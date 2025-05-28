# ffmpeg-split-encode

## Context

I’ve been transcoding videos to reduce disk usage for some time, but noticed that SVT-AV1 does not fully utilize available CPU cores, especially on my workstation. Initially, I manually split and encoded videos, but this script offers a cleaner, automated solution.

What it does is relatively simple:
- Read the original file and output some segments, audio is put aside for later
- Spawn FFmpeg instances based on the `affinities` to transcode all the above segments. Audio can be transcode as a whole but with some limitations
- Merge everything 

## Usage

Example:
```bash
python simple-svtav1-split-encode.py myvideo.mp4 "2@0*8"
```

| Argument   | Description |
|:-----------|:------------|
|`filename`    | the media file to encode |
|`affinities`  | comma separated cpu index or range of cpu indexes or cores-per-group@start[\*repeat], define parallelism (example: 0,1,2-3,4-7,2@8,2@10\*3) |

> _Note: for example `2@10*3` means 3 groups with 2 cores each, starting at core 10. It is equivalent to `10-11,12-13,14-15`._

| Option                            | Description  |
|:----------------------------------|:-------------|
|`-h`, `--help`                         | show this help message and exit|
|`--ffmpeg_path` FFMPEG_PATH          | path to ffmpeg executable (default expect ffmpeg to be in PATH)|
|`--ffprobe_path` FFPROBE_PATH        | path to ffprobe executable (default expect ffprobe to be in PATH)|
|`-d`, `--temp_dir` TEMP_DIR            | path to directory to store video fragments (default same directory as filename)|
|`-ss`, `--start_time` START_TIME       | start transcoding at specified time (same as ffmpeg)|
|`-to`, `--end_time` END_TIME           | stop transcoding after specified time is reached (same as ffmpeg)|
|`-t`, `--segment_time` SEGMENT_TIME    | segment duration in seconds (default = 5)|
|`-p`, `--params` PARAMS                | parameters for SVT-AV1 encoder (lp level is automatically set based on CPU affinities, default is 'preset=6:tune=0:keyint:10s:crf=45')
|`-c`, `--audio_codec` AUDIO_CODEC      | re-encode audio with codec (default is to copy audio from source file)|
|`-b`, `--audio_bitrate` AUDIO_BITRATE  | audio bitrate (default is audio encoder default bitrate)|
|`-o`, `--output` OUTPUT                | output file path (default is the same as input file with '-concat' suffix and '.mkv' extension)|

## Dependencies

On system:
- FFmpeg

For python:
- psutil
- tqdm

## Performance comparison

Here you can find some one-shot times to encode [bbb_sunflower_1080p_60fps_normal.mp4](https://download.blender.org/demo/movies/BBB/bbb_sunflower_1080p_60fps_normal.mp4.zip)

### Comparison Across CPUs

The followings are done with 10bits encodings and with these SVT-AV1 parameters: `preset=6:tune=0:keyint=10s:crf=45`

The baseline is the equivalent command with ffmpeg: 
```bash
ffmpeg -i ./bbb_sunflower_1080p_60fps_normal.mp4 -c:a copy -pix_fmt yuv420p10le -c:v libsvtav1 -svtav1-params preset=6:tune=0:keyint=10s:crf=45 ./out.mkv
```

| Configuration     | Ryzen 9 7950X3D     | 2× Xeon E5-2690v4     | i7-9700           | Ryzen 7 7840U     |
|:------------------|--------------------:|----------------------:|------------------:|------------------:|
| **Baseline**      | 304.33 (1.00×)      | 813.53 (1.00×)        | 696.98 (1.00×)    | 657.02 (1.00×)    |
| **2C/encoder**    | 206.20 (1.48×)      | 314.89 (2.58×)        | 632.74 (1.10×)    | 621.38 (1.06×)    |
| **1C/encoder**    | 188.74 (1.61×)      | 294.72 (2.76×)        | 569.84 (1.22×)    | 602.10 (1.09×)    |
| **1T/encoder**    | 180.07 (1.69×)      | 308.22 (2.64×)        | -                 | 565.00 (1.16×)    |

> _Note1: Times are in seconds and are followed by speedup relative to the baseline for that CPU._

> _Note2: `C` and `T` refer respectively to physical and logical processors, 1C = 2T for most consumer CPUs. Since i7-9700 has no HT, the last test is not possible._

My favourite setup is 1 encoder per physical core, one of the reason is the already (very) high amount of RAM it use.

### Cross-Preset Comparison on Ryzen 9 7950X3D

<table cellspacing="0" cellpadding="6">
  <thead>
    <tr>
      <th rowspan="2"></th>
      <th colspan="2" style="text-align: center;">Time (s)</th>
      <th colspan="2" style="text-align: center;">Size (KB)</th>
      <th colspan="2" style="text-align: center;">PSNR</th>
    </tr>
    <tr>
      <th>Baseline</th>
      <th>1C/encoder</th>
      <th>Baseline</th>
      <th>1C/encoder</th>
      <th>Baseline</th>
      <th>1C/encoder</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Preset 8</strong></td>
      <td style="text-align: right;">145.40</td>
      <td style="text-align: right;">104.47 (x1.39)</td>
      <td style="text-align: right;">128,548</td>
      <td style="text-align: right;">130,482</td>
      <td style="text-align: right;">42.95</td>
      <td style="text-align: right;">43.15</td>
    </tr>
    <tr>
      <td><strong>Preset 6</strong></td>
      <td style="text-align: right;">304.33</td>
      <td style="text-align: right;">188.74 (x1.61)</td>
      <td style="text-align: right;">126,537</td>
      <td style="text-align: right;">127,804</td>
      <td style="text-align: right;">43.94</td>
      <td style="text-align: right;">44.05</td>
    </tr>
    <tr>
      <td><strong>Preset 4</strong></td>
      <td style="text-align: right;">883.91</td>
      <td style="text-align: right;">558.92 (x1.58)</td>
      <td style="text-align: right;">129,418</td>
      <td style="text-align: right;">130,208</td>
      <td style="text-align: right;">44.55</td>
      <td style="text-align: right;">44.66</td>
    </tr>
  </tbody>
</table>

Since the script introduce some overhead, it works better if the tasks are not too short. Here the preset 6 and 4 offer a better speedup compared to preset 8. Size and PSNR values are close, with differences likely caused by segment durations (ranging from 1.284 to 8.533 seconds). Since keyint is 10 seconds, shorter segments may slightly increase the number of I-frames.

As the encodings complete, there are cases where only a few remain. This situation can reduce the average efficiency as the core affinities are fixed. This effect can be reduce either with less encoders (encode a segment faster) or smaller segments (spend less time per segment), but it tends to reduce average efficiency, making it a tradeoff worth considering. As a performance optimization, segments are sorted by duration (in descending order) before processing, reducing idle time at the end when only a few segments remain.

## Todo
- A way to stop and resume the encoding
- Avoid the segmentation by finding a way to reuse the original file with -ss and -t/-to but my current method might introduce some duplicate frames
- Use scene change information for better segmentation, has the same issue as above (help is welcome)
