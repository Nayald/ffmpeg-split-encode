import argparse
import subprocess
import psutil
import tqdm
import re

from pathlib import Path
from math import log2
from queue import Queue, Empty
from threading import Thread, Event
from concurrent.futures import ThreadPoolExecutor


"""
This script segments a video file into smaller parts based on scene changes, encodes each segment in parallel using the SVT-AV1 encoder, and then merges the encoded segments back into a single video file.
It uses ffmpeg for segmentation and encoding, and ffprobe to get the duration of the media file.
It supports specifying CPU affinities for parallel encoding, and can re-encode audio with a specified codec and bitrate.
It requires ffmpeg and ffprobe to be installed.
"""


def cleanup(path: Path) -> None:
    """
    Cleans up the temporary directory by removing all files and the directory itself.
    """
    for child in path.iterdir():
        child.unlink()

    path.rmdir()


def parse_affinity_specs(expr: str) -> list[tuple[int, int]]:
    """
    Parses core specs like:
    - single core: 0
    - range: 2-5
    - group with optional repeat: 2@4 or 2@4*6, where 2@4 means 2 cores starting from core 4, and 2@4*6 means 6 repetitions of that group.
    Raises ValueError if the input is invalid.
    """
    pattern = re.compile(r'(\d+)(?:-(\d+)|@(\d+)(\*\d+)?)?')
    affinities = []
    for part in expr.split(','):
        part = part.strip()
        m = pattern.fullmatch(part)
        if not m:
            raise ValueError(f"Invalid core spec: '{part}'")

        if m.group(2):  # Range: X-Y
            start = int(m.group(1))
            end = int(m.group(2))
            if end < start:
                raise ValueError(f"Invalid range: {part}")
            
            affinities.append((start, end))
        elif m.group(3):  # Group pattern: X@Y[*Z]
            count = int(m.group(1))
            start = int(m.group(3))
            repeat = int(m.group(4)[1:]) if m.group(4) else 1
            for i in range(repeat):
                base = start + i * count
                affinities.append((base, base + count - 1))
        else:  # Single core
            core = int(m.group(1))
            affinities.append((core, core))

    affinities.sort(key=lambda x: x[1] - x[1])
    return affinities


def parse_sexagesimal_time(time: str) -> float:
    """
    Parses a sexagesimal time string (e.g., "01:23:45") and converts it to seconds.
    Returns 0 if the input is "N/A" or invalid.
    """
    result = 0
    if time == "N/A":
        return result

    tokens = time.split(":")
    if len(tokens) > 3:
        return result
    
    for i, t in enumerate(reversed(tokens)):
        result += float(t) * 60 ** i
        
    return result


def get_duration(ffprobe_path: Path, media_path: Path) -> float:
    """
    Uses ffprobe to get the duration of the media file in seconds.
    Returns 0 if the media file is invalid or ffprobe fails.
    """
    args = [ffprobe_path]
    args += "-v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1".split()
    args.append(media_path)
    #print(*args)
    return float(subprocess.run(args, capture_output=True, text=True).stdout)


def segment(ffmpeg_path: Path, media_path: Path, output_dir: Path, segment_time, start=None, end=None) -> tuple[list[Path], Path]:
    """
    Segments the media file into smaller parts using ffmpeg.
    Returns a list of segment file paths and the path to the audio file.
    If the segmentation fails, returns an empty list and an empty Path.
    """
    args = [ffmpeg_path]
    args += "-hide_banner -nostats -y".split()
    if start:
        args.append("-ss")
        args.append(start)
    
    if end:
        args.append("-to")
        args.append(end)

    args.append("-i")
    args.append(media_path)
    args += "-an -c copy -f segment -reset_timestamps 1 -segment_time".split()
    args.append(segment_time)
    args.append(output_dir / "segment-%04d.mkv")
    args += "-vn -c:a copy ".split()
    audio_path = output_dir / "audio.mkv"
    args.append(audio_path)
    #print(*args)
    process = subprocess.run(args, capture_output=True, text=True, universal_newlines=True)
    if process.returncode != 0:
        return [], Path()
    
    return list(map(Path, re.findall(r"^\[segment @ \w*?\] Opening '(.+?)' for writing$", process.stderr, flags=re.MULTILINE))), audio_path


def encode(ffmpeg_path: Path, affinity: tuple[int, int], media_path: str | Path, output_path: Path | str | None = None, params: str | None = None) -> subprocess.Popen[str]:
    """
    Encodes the media file using SVT-AV1 encoder with specified parameters.
    Returns a subprocess.Popen object for the encoding process.
    If output_path is not specified, it defaults to "encoded-" + media_path name.
    If params is not specified, it defaults to "preset=6:tune=0:crf=45".
    """
    if isinstance(media_path, str):
        media_path = Path(media_path)

    if not output_path:
        output_path = media_path.with_name("encoded-" + media_path.name)

    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    if params is None:
        params = ":".join(f"{k}={v}" for k, v in (
            ("preset", 6),
            ("tune", 0),
            ("keyint", "10s"),
            ("crf", 45),
        ))

    args = [ffmpeg_path]
    args += "-v warning -nostats -progress pipe:1 -stats_period 1 -y -i".split()
    args.append(media_path)
    # log2 is almost like lp level in https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/master/Source/Lib/Globals/enc_handle.c
    args += f"-an -pix_fmt yuv420p10le -c:v libsvtav1 -svtav1-params {params}:lp={log2(affinity[1] - affinity[0] + 2):.0f}".split()
    args.append(output_path)
    #print(*args)
    process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, universal_newlines=True)
    psutil.Process(process.pid).cpu_affinity(list(range(affinity[0], affinity[1] + 1)))
    return process

    
stop_event = Event()
def process_encoding_queue(ffmpeg_path: Path, ffprobe_path: Path, affinity: tuple[int, int], global_pbar: tqdm.tqdm, encoding_queue: Queue[tuple[Path, float, Path, str]]) -> None:
    """
    Processes the encoding queue, encoding media files using the specified ffmpeg path and affinity.
    Updates the global progress bar and handles termination of the encoding process if stop_event is set.
    """
    while not stop_event.is_set():
        try:
            media_path, segment_duration, output_path, params = encoding_queue.get_nowait()
        except Empty:
            break
        
        process = encode(ffmpeg_path, affinity, media_path, output_path, params)
        with tqdm.tqdm(total=int(1_000_000 * segment_duration), desc=f"[T{affinity[0]:<2}-{affinity[1]:>2}] {media_path.name}", unit="µsec(s)", leave=False) as pbar:
            # read the stream until eof is reached
            for line in iter(process.stdout.readline, ''):
                if stop_event.is_set():
                    process.terminate()
                    break

                match line.rstrip().split("=", 1):
                    case ("fps", fps):
                        pbar.set_postfix(fps=fps)
                    case ("out_time_us", time) if time != "N/A":
                        pbar.update(int(time) - pbar.n)

        try:
            process.wait(15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            
        global_pbar.update()
        encoding_queue.task_done()
        media_path.unlink(missing_ok=True)
        
        
def encode_audio(ffmpeg_path: Path, media_path: str | Path, audio_codec: str, audio_bitrate: str | None = None, output_path: Path | str | None = None) -> subprocess.Popen[str]:
    """
    Encodes the audio of the media file using the specified audio codec and bitrate.
    Returns a subprocess.Popen object for the encoding process.
    If output_path is not specified, it defaults to "encoded-" + media_path name.
    If audio_bitrate is not specified, it uses the default bitrate of the audio codec.
    """
    if isinstance(media_path, str):
        media_path = Path(media_path)

    if not output_path:
        output_path = media_path.with_name("encoded-" + media_path.name)

    if isinstance(output_path, str):
        output_path = Path(output_path)

    args = [ffmpeg_path]
    args += "-v warning -nostats -progress pipe:1 -stats_period 1 -y -i".split()
    args.append(media_path)
    args += "-vn -c:a".split()
    args.append(audio_codec)
    if audio_bitrate:
        args.append("-b:a")
        args.append(audio_bitrate)
        
    args.append(output_path)
    #print(*args)
    process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, universal_newlines=True, creationflags=subprocess.DETACHED_PROCESS)
    return process


def concatenate(ffmpeg_path: Path, segment_paths: list[Path], audio_path: Path, output_path: Path) -> bool:
    """
    Concatenates the video segments and audio file into a single output file using ffmpeg.
    Returns True if the concatenation is successful, False otherwise.
    """
    concat_file = temp_dir / "concat.txt"
    with open(concat_file, mode="w") as f:
        f.writelines(f"file '{segment_path.absolute()}'\n" for segment_path in segment_paths)

    args = [ffmpeg_path]
    args += f"-v warning -stats -y -f concat -safe 0 -i".split()
    args.append(concat_file)
    args.append("-i")
    args.append(audio_path)
    args += "-map 0:v -map 1:a -c copy".split()
    args.append(output_path)
    #print(*args)
    return subprocess.run(args).returncode == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segments a video file and encode them in parallel with SVT-AV1 encoders.")
    parser.add_argument("--ffmpeg_path", default="ffmpeg", help="path to ffmpeg executable (default expect ffmpeg to be in PATH)")
    parser.add_argument("--ffprobe_path", default="ffprobe", help="path to ffprobe executable (default expect ffprobe to be in PATH)")
    parser.add_argument("-d", "--temp_dir", type=Path, help="path to directory to store video fragments (default same directory as filename)")
    parser.add_argument("-ss", "--start_time", default=None, help="start transcoding at specified time (same as ffmpeg)")
    parser.add_argument("-to", "--end_time", default=None, help="stop transcoding after specified time is reached (same as ffmpeg)")
    parser.add_argument("-t", "--segment_time", default="5", help="segment duration in seconds (default = 5)")
    parser.add_argument("-p", "--params", help="parameters for SVT-AV1 encoder (lp level is automatically set based on CPU affinities, default is 'preset=6:tune=0:keyint:10s:crf=45')")
    parser.add_argument("-c", "--audio_codec", help="re-encode audio with codec (default is to copy audio from source file)")
    parser.add_argument("-b", "--audio_bitrate", help="audio bitrate (default is audio encoder default bitrate)")
    parser.add_argument("filename", type=Path, help="the media file to encode")
    parser.add_argument("affinities", help="comma separated cpu index or range of cpu indexes or cores-per-group@start[*repeat], define parallelism (example: 0,1,2-3,4-7,2@8,2@10*3)")
    options = parser.parse_args()

    if not options.filename.exists():
        exit(-1)

    affinities = parse_affinity_specs(options.affinities)
    print(f"concurrency is set to {len(affinities)}")
    if not options.temp_dir:
        temp_dir = options.filename.with_name(options.filename.name + "-temp")
    else:
        temp_dir = Path(options.temp_dir)

    temp_dir.mkdir(parents=True, exist_ok=True)
    segment_paths, audio_path = segment(options.ffmpeg_path, options.filename, temp_dir, options.segment_time, options.start_time, options.end_time)
    if not segment_paths:
        print("fail to segment file")
        cleanup(temp_dir)
        exit(-1)

    print(f"got {len(segment_paths)} segments, sorting by duration...")
    with ThreadPoolExecutor(max_workers=len(affinities)) as executor:
        segment_durations = list(tqdm.tqdm(executor.map(lambda x: get_duration(options.ffprobe_path, x), segment_paths), total=len(segment_paths), desc="getting segment durations", unit="segment(s)"))

    encoded_segment_paths = [segment_path.with_name("encoded-" + segment_path.name) for segment_path in segment_paths]
    encoding_queue = Queue()
    for segment_path, segment_duration, encoded_segment_path in sorted(zip(segment_paths, segment_durations, encoded_segment_paths), key=lambda x: x[1], reverse=True):
        encoding_queue.put((segment_path, segment_duration, encoded_segment_path, options.params))
        
    print("encoding segments...")
    with tqdm.tqdm(total=len(segment_paths), desc="segments done", unit="segment(s)", leave=True) as global_pbar:
        encode_threads = []
        for affinity in affinities:
            t = Thread(target=process_encoding_queue, args=(options.ffmpeg_path, options.ffprobe_path, affinity, global_pbar, encoding_queue))
            t.start()
            encode_threads.append(t)
        
        if options.audio_codec:
            encoded_audio_path = audio_path.with_name("encoded-" + audio_path.name)
            audio_process = encode_audio(options.ffmpeg_path, audio_path, options.audio_codec, options.audio_bitrate, encoded_audio_path)
            with tqdm.tqdm(total=int(1_000_000 * get_duration(options.ffprobe_path, audio_path)), desc=audio_path.name, unit="µsec(s)", leave=False) as audio_pbar:
                for line in iter(audio_process.stdout.readline, ''):
                    match line.rstrip().split("=", 1):
                        case ("out_time_us", time) if time != "N/A":
                            audio_pbar.update(int(time) - audio_pbar.n)
                
            try:
                audio_process.wait(15)
            except subprocess.TimeoutExpired:
                audio_process.kill()
                audio_process.wait()

            audio_path.unlink(missing_ok=True)
        else:
            encoded_audio_path = audio_path
        
        try:
            while any(t.is_alive() for t in encode_threads):
                for t in encode_threads:
                    t.join(1)
        except KeyboardInterrupt:
            stop_event.set()
            for t in encode_threads:
                t.join()
                
            global_pbar.write("threads stopped by user")
            cleanup(temp_dir)
            exit(-1)

    print("all segments encoded, merging...")
    if not concatenate(options.ffmpeg_path, encoded_segment_paths, encoded_audio_path, options.filename.with_name(options.filename.stem + "-concat.mkv")):
        print("fail to merge segments")
        cleanup(temp_dir)
        exit(-1)

    print("all segments merged, cleaning up...")
    cleanup(temp_dir)