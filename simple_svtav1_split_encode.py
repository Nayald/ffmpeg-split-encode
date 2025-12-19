# /// script
# dependencies = [
#     "psutil",
#     "tqdm",
# ]
# ///

import argparse
import subprocess
from pathlib import Path
from queue import Queue
from threading import Event, Thread

import tqdm

from encode_handlers import map_affinities_to_handlers

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
        child.unlink(missing_ok=True)

    path.rmdir()


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
        result += float(t) * 60**i

    return result


def get_duration(ffprobe_path: Path, media_path: Path) -> float:
    """
    Uses ffprobe to get the duration of the media file in seconds.
    Returns 0 if the media file is invalid or ffprobe fails.
    """
    args = [ffprobe_path, "-v", "quiet", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", media_path]
    # print(*args)
    return float(subprocess.run(args, capture_output=True, text=True).stdout)


def segment(ffmpeg_path: Path, media_path: Path, output_dir: Path, segment_time: float | str, start: str | None = None, end: str | None = None) -> tuple[list[tuple[Path, float]], Path]:
    """
    Segments the media file into smaller parts using ffmpeg.
    Returns a list of segment file paths and the path to the audio file.
    If the segmentation fails, returns an empty list and an empty Path.
    """
    args = [ffmpeg_path, "-v", "warning", "-nostats", "-y"]
    if start:
        args += ["-ss", start]

    if end:
        args += ["-to", end]

    audio_path = output_dir / "audio.mkv"
    args += ["-i", media_path]
    args += ["-an", "-c", "copy", "-f", "segment", "-reset_timestamps", "1", "-segment_time", segment_time, "-segment_list", "pipe:1", "-segment_list_type", "csv", output_dir / "segment-%04d.mkv"]
    args += ["-vn", "-c:a", "copy", audio_path]
    # print(*args)
    process = subprocess.run(args, capture_output=True, text=True)
    if process.returncode != 0:
        return [], Path()

    return [(output_dir / p, float(e) - float(s)) for p, s, e in (x.split(",") for x in process.stdout.splitlines())], audio_path


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

    args = [ffmpeg_path, "-v", "warning", "-nostats", "-progress", "pipe:1", "-stats_period", "1", "-y", "-i", media_path, "-vn", "-c:a", audio_codec]
    if audio_bitrate:
        args.append("-b:a")
        args.append(audio_bitrate)

    args.append(output_path)
    # print(*args)
    return subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, universal_newlines=True, creationflags=subprocess.DETACHED_PROCESS)


def concatenate(ffmpeg_path: Path, segment_paths: list[Path], audio_path: Path, output_path: Path) -> bool:
    """
    Concatenates the video segments and audio file into a single output file using ffmpeg.
    Returns True if the concatenation is successful, False otherwise.
    """
    concat_file = temp_dir / "concat.txt"
    with open(concat_file, mode="w") as f:
        f.writelines(f"file '{segment_path.absolute()}'\n" for segment_path in segment_paths)

    args = [ffmpeg_path, "-v", "warning", "-stats", "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-i", audio_path, "-map", "0:v", "-map", "1:a", "-c", "copy", output_path]
    # print(*args)
    return subprocess.run(args).returncode == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segments a video file and encode them in parallel with SVT-AV1 encoders.")
    parser.add_argument("-i", "--ssh_key", default=None, help="path to ssh private key for remote hosts (default is to use ssh default key)")
    parser.add_argument("--ffmpeg_path", default="ffmpeg", help="path to ffmpeg executable (default expect ffmpeg to be in PATH)")
    parser.add_argument("--ffprobe_path", default="ffprobe", help="path to ffprobe executable (default expect ffprobe to be in PATH)")
    parser.add_argument("-d", "--temp_dir", type=Path, help="path to directory to store video fragments (default same directory as filename)")
    parser.add_argument("-ss", "--start_time", default=None, help="start transcoding at specified time (same as ffmpeg)")
    parser.add_argument("-to", "--end_time", default=None, help="stop transcoding after specified time is reached (same as ffmpeg)")
    parser.add_argument("-t", "--segment_time", default="5", help="segment duration in seconds (default = 5)")
    parser.add_argument("-p", "--params", help="parameters for SVT-AV1 encoder (lp level is automatically set based on CPU affinities, default is 'preset=6:tune=0:keyint:10s:crf=45')")
    parser.add_argument("-c", "--audio_codec", help="re-encode audio with codec (default is to copy audio from source file)")
    parser.add_argument("-b", "--audio_bitrate", help="audio bitrate (default is audio encoder default bitrate)")
    parser.add_argument("-o", "--output", type=Path, help="output file path (default is the same as input file with '-concat' suffix and '.mkv' extension)")
    parser.add_argument("filename", type=Path, help="the media file to encode")
    parser.add_argument(
        "affinities",
        nargs="+",
        help=(
            "comma separated cpu index or range of cpu indexes or cores-per-group@start[*repeat], define parallelism (example: 0,1,2-3,4-7,2@8,2@10*3)"
            ", can also specify remote host by prefixing with hostname= (example: 192.168.1.2=0,1-2,1@3*2), can be repeated to define multiple hosts"
        ),
    )
    options = parser.parse_args()

    if not options.filename.exists():
        print("filename does not exist")
        exit(-1)

    stop_event = Event()
    process_handlers = map_affinities_to_handlers(options.affinities, options.ffmpeg_path, options.ssh_key, stop_event)
    print("concurrency is set to", len(process_handlers))
    temp_dir = options.filename.with_name(options.filename.stem + "-tmp") if not options.temp_dir else Path(options.temp_dir)

    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path = options.filename.with_name(options.filename.stem + "-concat.mkv") if not options.output else options.output

    segment_infos, audio_path = segment(options.ffmpeg_path, options.filename, temp_dir, options.segment_time, options.start_time, options.end_time)
    if not segment_infos:
        print("fail to segment file")
        cleanup(temp_dir)
        exit(-1)

    encoded_segment_paths = [segment_info[0].with_name("encoded-" + segment_info[0].name) for segment_info in segment_infos]
    encoding_queue = Queue()
    for segment_path, segment_duration, encoded_segment_path in sorted(zip(*zip(*segment_infos, strict=True), encoded_segment_paths, strict=True), key=lambda x: x[1], reverse=True):
        encoding_queue.put((segment_path, segment_duration, encoded_segment_path, options.params, 3))

    print("encoding segments...")
    with tqdm.tqdm(total=len(segment_infos), desc="segments done", unit="segment(s)", leave=True) as global_pbar:
        encode_threads = []
        for process_handler in process_handlers:
            t = Thread(target=process_handler.run, args=(global_pbar, encoding_queue))
            t.start()
            encode_threads.append(t)

        if options.audio_codec:
            encoded_audio_path = audio_path.with_name("encoded-" + audio_path.name)
            audio_process = encode_audio(options.ffmpeg_path, audio_path, options.audio_codec, options.audio_bitrate, encoded_audio_path)
            with tqdm.tqdm(total=int(get_duration(options.ffprobe_path, audio_path) * 1000), desc=audio_path.name, unit="ms", leave=False) as audio_pbar:
                assert audio_process.stdout is not None
                for line in iter(audio_process.stdout.readline, ""):
                    match line.rstrip().split("=", 1):
                        case ("out_time_us", time) if time != "N/A":
                            audio_pbar.update(int(time) // 1000 - audio_pbar.n)

            try:
                audio_process.wait(15)
            except subprocess.TimeoutExpired:
                audio_process.kill()
                audio_process.wait()

            audio_path.unlink(missing_ok=True)
        else:
            encoded_audio_path = audio_path

        try:
            encoding_queue.join()
            stop_event.set()
            for t in encode_threads:
                t.join()
        except KeyboardInterrupt:
            stop_event.set()
            for t in encode_threads:
                t.join()

            global_pbar.write("threads stopped by user")
            cleanup(temp_dir)
            exit(-1)

    print("all segments encoded, merging...")
    if not concatenate(options.ffmpeg_path, encoded_segment_paths, encoded_audio_path, output_path):
        print("fail to merge segments")
        cleanup(temp_dir)
        exit(-1)

    print("all segments merged, cleaning up...")
    cleanup(temp_dir)
