import re
import subprocess
from abc import ABC, abstractmethod
from io import BufferedReader, BufferedWriter, TextIOWrapper
from math import log2
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

import psutil
import tqdm


class ProcessHandler(ABC):
    def __init__(self, affinity: tuple[int, int], stop_event: Event | None = None) -> None:
        self.affinity = affinity
        self.stop_event = stop_event if stop_event else Event()

    @abstractmethod
    def run(self, global_pbar: tqdm.tqdm, encoding_queue: Queue[tuple[Path, float, Path, str, int]]) -> None:
        pass


class HostProcessHandler(ProcessHandler):
    def __init__(self, ffmpeg_path: Path, affinity: tuple[int, int], stop_event: Event | None = None) -> None:
        super().__init__(affinity=affinity, stop_event=stop_event)
        self.ffmpeg_path = ffmpeg_path

    def create_process(self, media_path: Path | str, output_path: Path | str, params: str) -> subprocess.Popen[str]:
        args = [self.ffmpeg_path]
        args += ["-v", "warning", "-nostats", "-progress", "pipe:1", "-stats_period", "1", "-y", "-i"]
        args.append(media_path)
        # log2 is almost like lp level in https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/master/Source/Lib/Globals/enc_handle.c
        args += f"-an -pix_fmt yuv420p10le -c:v libsvtav1 -svtav1-params {params}:lp={log2(self.affinity[1] - self.affinity[0] + 2):.0f}".split()
        args.append(output_path)

        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, universal_newlines=True)
        psutil.Process(process.pid).cpu_affinity(list(range(self.affinity[0], self.affinity[1] + 1)))
        return process

    def run(self, global_pbar: tqdm.tqdm, encoding_queue: Queue[tuple[Path, float, Path, str, int]]) -> None:
        """
        Processes the encoding queue, encoding media files using the specified ffmpeg path and affinity.
        Updates the global progress bar and handles termination of the encoding process if stop_event is set.
        """
        while not self.stop_event.is_set():
            try:
                media_path, segment_duration, output_path, params, retries = encoding_queue.get(timeout=0.1)
            except Empty:
                continue

            if retries <= 0:
                raise RuntimeError(f"Encoding failed too many times for {media_path}")

            if isinstance(media_path, str):
                media_path = Path(media_path)

            if not output_path:
                output_path = media_path.with_name("encoded-" + media_path.name)

            if isinstance(output_path, str):
                output_path = Path(output_path)

            if params is None:
                params = "preset=6:tune=0:keyint=10s:crf=45"

            p = self.create_process(media_path, output_path, params)
            with tqdm.tqdm(total=int(segment_duration * 1000), desc=f"[T{self.affinity[0]:<2}-{self.affinity[1]:>2}] {media_path.name}", unit="ms", leave=False) as pbar:
                # read the stream until eof is reached
                assert p.stdout is not None
                for line in iter(p.stdout.readline, ""):
                    if self.stop_event.is_set():
                        p.terminate()
                        break

                    match line.rstrip().split("=", 1):
                        case ("fps", fps):
                            pbar.set_postfix(fps=fps)
                        case ("out_time_us", time) if time != "N/A":
                            pbar.update(int(time) // 1000 - pbar.n)

            try:
                p.wait(15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()

            if p.returncode == 0:
                global_pbar.update()
                encoding_queue.task_done()
                media_path.unlink(missing_ok=True)
            else:
                global_pbar.write(f"[{self}] Encoding failed for {media_path} with return code {p.returncode}. Retries left: {retries - 1}")
                encoding_queue.put((media_path, segment_duration, output_path, params, retries - 1))
                encoding_queue.task_done()

    def __repr__(self) -> str:
        return f"HostProcessHandler(affinity={self.affinity})"


class RemoteProcessHandler(ProcessHandler):
    def __init__(self, host: str, affinity: tuple[int, int], ssh_key: Path | str | None = None, stop_event: Event | None = None) -> None:
        super().__init__(affinity=affinity, stop_event=stop_event)
        self.host = host
        self.ssh_key = ssh_key

    def send_data(self, stdin: BufferedWriter, media_path: Path | str) -> None:
        with open(media_path, "rb") as f:
            try:
                while not self.stop_event.is_set() and (d := f.read(1_048_576)):
                    stdin.write(d)

                stdin.close()
            except Exception as e:
                tqdm.tqdm.write(f"[{self}], stdin: {e}")

    def receive_data(self, stdout: BufferedReader, output_path: Path | str) -> None:
        with open(output_path, "wb") as f:
            try:
                while not self.stop_event.is_set() and (d := stdout.read(1_048_576)):
                    f.write(d)

            except Exception as e:
                tqdm.tqdm.write(f"[{self}] stdout: {e}")

    def receive_stats(self, stderr: BufferedReader, pbar: tqdm.tqdm, global_pbar: tqdm.tqdm) -> None:
        with TextIOWrapper(stderr, encoding="utf-8", errors="replace") as stream:
            try:
                while not self.stop_event.is_set() and (line := stream.readline()):
                    match line.rstrip().split("=", 1):
                        case ("fps", fps):
                            pbar.set_postfix(fps=fps)
                        case ("out_time_us", time) if time != "N/A":
                            pbar.update(int(time) // 1000 - pbar.n)
            except Exception as e:
                tqdm.tqdm.write(f"[{self}], stderr: {e}")

    def create_process(self, params: str) -> subprocess.Popen[bytes]:
        args = ["ssh"]
        if self.ssh_key:
            args += ["-i", self.ssh_key]

        args.append(self.host)
        args.append(
            f"taskset -c {self.affinity[0]}-{self.affinity[1]} ffmpeg -v warning -nostats -progress pipe:2 -stats_period 1 -y -i - \
                -an -pix_fmt yuv420p10le -c:v libsvtav1 -svtav1-params {params}:lp={log2(self.affinity[1] - self.affinity[0] + 2):.0f} -f matroska -"
        )
        return subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def run(self, global_pbar: tqdm.tqdm, encoding_queue: Queue[tuple[Path, float, Path, str, int]]) -> None:
        """
        Processes the encoding queue, encoding media files using the specified ffmpeg path and affinity.
        Updates the global progress bar and handles termination of the encoding process if stop_event is set.
        """
        while not self.stop_event.is_set():
            try:
                media_path, segment_duration, output_path, params, retries = encoding_queue.get(timeout=0.1)
            except Empty:
                continue

            if retries <= 0:
                raise RuntimeError(f"Encoding failed too many times for {media_path}")

            if isinstance(media_path, str):
                media_path = Path(media_path)

            if not output_path:
                output_path = media_path.with_name("encoded-" + media_path.name)

            if isinstance(output_path, str):
                output_path = Path(output_path)

            if params is None:
                params = "preset=6:tune=0:keyint=10s:crf=45"

            p = self.create_process(params)
            send_data_thread = Thread(target=self.send_data, args=(p.stdin, media_path))
            send_data_thread.start()
            receive_data_thread = Thread(target=self.receive_data, args=(p.stdout, output_path))
            receive_data_thread.start()
            pbar = tqdm.tqdm(total=int(segment_duration * 1000), desc=f"[{self.host}:T{self.affinity[0]:<2}-{self.affinity[1]:>2}] {media_path.name}", unit="ms", leave=False)
            receive_stats_thread = Thread(target=self.receive_stats, args=(p.stderr, pbar, global_pbar))
            receive_stats_thread.start()

            send_data_thread.join()
            receive_data_thread.join()
            receive_stats_thread.join()

            try:
                p.wait(15) if not self.stop_event.is_set() else p.terminate()
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()

            if p.returncode == 0:
                global_pbar.update()
                encoding_queue.task_done()
                # media_path.unlink(missing_ok=True)
            else:
                global_pbar.write(f"[{self}] Encoding failed for {media_path} with return code {p.returncode}. Retries left: {retries - 1}")
                encoding_queue.put((media_path, segment_duration, output_path, params, retries - 1))
                encoding_queue.task_done()

    def __repr__(self) -> str:
        return f"RemoteProcessHandler(host={self.host}, affinity={self.affinity})"


def parse_affinity_specification(expr: str) -> list[tuple[int, int]]:
    """
    Parses core specs like:
    - single core: 0
    - range: 2-5
    - group with optional repeat: 2@4 or 2@4*6, where 2@4 means 2 cores starting from core 4, and 2@4*6 means 6 repetitions of that group.
    Raises ValueError if the input is invalid.
    """
    pattern = re.compile(r"(\d+)(?:-(\d+)|@(\d+)(\*\d+)?)?")
    affinities = []
    for part in expr.split(","):
        part = part.strip()
        m = pattern.fullmatch(part)
        if not m:
            raise ValueError(f"Invalid core specification: '{part}'")

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

    # affinities.sort(key=lambda x: x[1] - x[1])
    return affinities


def map_affinities_to_handlers(affinities: list[str], ffmpeg_path: Path, ssh_key: Path | str | None = None, stop_event: Event | None = None) -> list[ProcessHandler]:
    """
    Maps a list of CPU affinities to corresponding ProcessHandler instances.
    If host is specified, RemoteProcessHandler is used; otherwise, HostProcessHandler is used.
    Args:
        affinities (list[str]): List of affinity specifications, e.g., ["0-3", "remotehost=4-7"].
        ffmpeg_path (Path): Path to the ffmpeg executable for local processing.
        ssh_key (Path | str | None): Optional SSH key path for remote processing.
        stop_event (Event | None): Optional threading event to signal stopping.
    """
    handlers = []
    for affinity in affinities:
        pos = affinity.find("=")
        handlers.extend(
            RemoteProcessHandler(affinity[:pos], a, ssh_key, stop_event) if pos != -1 else HostProcessHandler(ffmpeg_path, a, stop_event) for a in parse_affinity_specification(affinity[pos + 1 :])
        )

    handlers.sort(key=lambda h: (h.affinity[1] - h.affinity[0]) << isinstance(h, HostProcessHandler), reverse=True)  # small favor for local handlers
    return handlers
