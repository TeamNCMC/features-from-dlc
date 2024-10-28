"""
video_cutter command line interface (CLI).

author : Guillaume Le Goc (g.legoc@posteo.org), Edwin Gatier
version : 2024.08.30
"""

import os
import re
import subprocess
from typing import Optional

import numpy as np
import pandas as pd
import typer
from tqdm import tqdm
from typing_extensions import Annotated

__version__ = "2024.08.30"

# --- Parameters
TIME_BEFORE_ONSET = 0.5  # time before onset (seconds)
TIME_AFTER_ONSET = 1.5  # time after onset (seconds)

LASER_MIN_VOLTAGE = 1.5  # min. voltage for the laser to be "on" (volt)
CAMERA_MIN_VOLTAGE = 1.5  # min. voltage for the camera to be "on" (for bin files, volt)
SAMPLE_RATE = 10000  # sampling rate (for .bin files)
ADJUST_FACTOR = -0.000035  # fix delay between camera and laser

# path to the bin directory of ffmpeg, or empty string if in PATH
FFMPEG_BIN_DIR = ""

# separator in the laser file (only for txt or csv). Usually, "\t" for labscribe, ","
# for Fiji.
SEP = "\t"

# files extensions of videos and laser trace (without the dot)
VIDEO_EXT = "mp4"
LASER_EXT = "txt"

# --- Preparation
# this thresold is the value of the diff of the gradient of times when the laser is "on"
# above which we consider it is a different trial. Unless you do weird protocols where
# the time between two trials is of the same order of magnitude as the laser frequency,
# the default value should work (0.1).
TTHRESHOLD = 0.1


# --- Typer functions
def laser_color_callback(value: str):
    if value not in ("blue", "orange", None):
        raise typer.BadParameter("Choose either 'blue' or 'orange'")


def laser_ext_callback(value: str):
    if value not in ("txt", "csv", "bin"):
        raise typer.BadParameter("Laser files must be in .txt, .csv or .bin")


def version_callback(value: bool):
    if value:
        print(f"video-cutter CLI version : {__version__}")
        raise typer.Exit()


# --- Processing functions
def get_ffmpeg_exe(ffmpeg_bin_dir: str) -> tuple[str, str]:
    """
    Get the ffmpeg executable.

    Parameters
    ----------
    ffmpeg_bin_dir : str
        Full path to the bin directory of ffmpeg, or empty string if it is in the PATH.

    Returns
    -------
    ffprobe_path, ffmpeg_path : str

    """
    if len(ffmpeg_bin_dir) == 0:
        ffprobe_path = "ffprobe"
        ffmpeg_path = "ffmpeg"
    else:
        ffprobe_path = os.path.join(ffmpeg_bin_dir, "ffprobe")
        ffmpeg_path = os.path.join(ffmpeg_bin_dir, "ffmpeg")

    return ffprobe_path, ffmpeg_path


def find_video_duration(videofile: str, ffprobe_path: str) -> float:
    """
    Get video duration with ffprobe.

    Parameters
    ----------
    videofile : str
        Full path to video file.
    ffprobe_path : str
        Path to ffprobe command.

    Returns
    -------
    duration : float
        Video duration in seconds.

    """
    return float(
        subprocess.run(
            [
                ffprobe_path,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                videofile,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout
    )


def cut_video(
    ffmpeg_path: str, video: str, newvideo: str, start: float, duration: float
):
    """
    Extract portion of video with ffmpeg.

    Parameters
    ----------
    ffmpeg_path : str
        Full path to ffmpeg executable.
    video : str
        Full path to original video.
    newvideo : str
        Full path to written video.
    start : float
        Where to start the new video in the original one (in seconds).
    duration : float
        Duration of the new video (in seconds).

    """
    out = subprocess.run(
        [
            ffmpeg_path,
            "-ss",
            str(start),
            "-i",
            video,
            "-c:v",
            "h264",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-t",
            str(duration),
            newvideo,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    return out.returncode


def find_videos_with_laser(
    videos_directory: str, videoext: str = "mp4", laserext: str = "txt"
) -> list[str]:
    """
    Find videos and corresponding text files in the input directory.

    This returns a list of base name of each files (without extension).

    Parameters
    ----------
    videos_directory : str
        Direcotry to scan.
    videoext : str, optional
        Video files extension, by default "mp4".
    laserext : str, optional
        Laser files extension, by default "txt".

    Returns
    -------
    files_list : list[str]
        List of filenames without extension.

    """
    # list video files
    video_files = [
        os.path.splitext(filename)[0]
        for filename in os.listdir(videos_directory)
        if filename.endswith(videoext)
    ]
    # list laser files
    laser_files = [
        os.path.splitext(filename)[0]
        for filename in os.listdir(videos_directory)
        if filename.endswith(laserext)
    ]

    # return lits of items that appear in the two list
    return list(set(video_files) & set(laser_files))


def check_text_header(filepath: str) -> bool:
    """
    Check if the input file has a header, eg. column names or is there data directly.

    This is done by removing common separation (tab and comma), ignoring newline, and
    check if the data is only numeric : in that case, it returns False (no header).

    Parameters
    ----------
    filepath : str
        Full path to the file to analyze.

    Returns
    -------
    header : bool
        True if there is an header, False otherwise (numeric data on the first line).

    """
    # regexp pattern to find numeric data
    number_pattern = re.compile(r"^\s*-?\d+(\.\d+)?\s*$")
    # read first line
    with open(filepath) as f:
        first_line = f.readline().strip()
    # split by separators
    parts = re.split(r"[\t,]", first_line)

    # check all parts
    res = []
    for part in parts:
        if not number_pattern.match(part.strip()):
            res.append(True)
        else:
            res.append(False)

    return np.any(res)


def read_txt_file(
    laserfile: str, sep: str = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read laser txt file with pandas. The time must be on the fist column, signal on the
    second.

    Parameters
    ----------
    laserfile : str
        Full path to the laser file.
    sep : str or None, optional
        Separator used in the file, by default None (infered from file).

    Returns
    -------
    time : np.ndarray
    laser : np.ndarray

    """
    # determine if the file has a header, eg. column names on first row
    if check_text_header(laserfile):
        # header
        df = pd.read_csv(laserfile, sep=sep, usecols=[0, 1])
    else:
        # no header
        df = pd.read_csv(laserfile, sep=sep, header=None, usecols=[0, 1])

    time = df.iloc[:, 0].to_numpy()
    laser = df.iloc[:, 1].to_numpy()

    return time, laser


def read_bin_file(
    laserfile: str,
    color: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read laser bin file from Bonsai. Returns the camera and laser traces.

    This file must be created with Bonsai workflow : a binary file with blue laser data
    on the first column and orange laser data on the second. The time is created from
    the specified sample rate.

    Parameters
    ----------
    laserfile : str
        Full path to the bin laser file.
    color : {"blue", "orange"}
        Color to use to crop videos, "orange" or "blue".

    Returns
    -------
    camera : np.ndarray
    laser : np.ndarray

    """
    # load data
    data = np.fromfile(laserfile, np.float64).reshape(-1, 3)
    # get camera time trace
    camera = data[:, 2]
    # get laser time trace
    if color == "blue":
        laser = data[:, 0]
    elif color == "orange":
        laser = data[:, 1]
    else:
        raise ValueError(
            f"Did you specify laser color correctly ? Got {color}, expected 'blue' or 'orange'."
        )

    return camera, laser


def align_laser_video(
    camera: np.ndarray,
    laser: np.ndarray,
    camera_min_voltage: float,
    sample_rate: float,
    adjust_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect when the first frame is taken to crop the laser trace so that it is syncd
    with the video.

    Parameters
    ----------
    camera : np.ndarray
        Camera exposure voltage trace.
    laser : np.ndarray
        Laser voltage trace.
    camera_min_voltage : float
        Minimum camera exposure voltage to consider the camera is taking a frame (volt).
    sample_rate : float
        Traces sampling rate (set in Bonsai).
    adjust_factor : float
        Shrinking (if <0) or expanding (>0) factor for time.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        time, laser

    """
    # find index of first frame (eg. beginnning of the video)
    first_frame_idx = np.argmax(camera >= camera_min_voltage)
    # crop the laser trace
    laser = laser[first_frame_idx:]
    time = np.linspace(0, (len(laser) - 1) / sample_rate, len(laser)) * (
        1 + adjust_factor
    )

    return time, laser


def find_stim_onsets(
    time: np.ndarray, laser: np.ndarray, min_voltage: float, threshold: float
) -> np.ndarray:
    """
    Find stimulation onsets.

    Parameters
    ----------
    time : np.ndarray
        Time vector.
    laser : np.ndarray
        Laser voltage vector.
    min_voltage : float
        Minimum voltage for the laser to be "on".
    threshold : float
        Time diff. threshold to separate trials from laser pulses.

    Returns
    -------
    onset_times : np.ndarray
        List of stimulation onsets in same units as `time`.

    """
    # times where laser > min_voltage
    times_above = time[laser > min_voltage]
    # find big jumps in those times to find groups of peaks
    # positive : offset, negative : onset
    jumps = np.diff(np.gradient(times_above))
    # pad 0 at the end so it is the same size for indexing
    jumps = np.hstack((jumps, 0))
    # first time where laser is up and all times where second diff is inferior than -threshold
    onset_times = np.hstack((times_above[0], times_above[jumps < -threshold]))

    return onset_times


def filter_stim_single_in_window(
    onset_times: np.ndarray,
    time_before_onset: float,
    time_after_onset: float,
    video_duration: float,
) -> np.ndarray:
    """
    Find stimulation that are single in the defined time window.

    Removes stimulation that can't be used because another stimulation happens in the
    requested time windows (eg. from `onset - time_before_stim` to
    `onset + time_after_stim`).

    Parameters
    ----------
    onset_times : np.ndarray
        Stimulation onsets.
    time_before_onset : float
        Time before stim onset to extract.
    time_after_onset : float
        Time after stim onset to extract.
    video_duration : float
        Duration of the full video.

    Returns
    -------
    np.ndarray
        List of usable time onsets to crop the full video.

    """
    # time between video start and first stim onset
    time_before_first = onset_times[0]
    # time between each stim
    times_between_stim = np.diff(onset_times)
    # time between last stim onset and video end
    time_after_last = video_duration - onset_times[-1]

    # time elapsed before each stim onsets without any other stim
    time_before_without = np.hstack((time_before_first, times_between_stim))
    # time elapsed after each stim onsets without any other stim
    time_after_without = np.hstack((times_between_stim, time_after_last))
    # select isolated stim, eg. that are single in the defined time window
    final_onsets = onset_times[
        (time_before_without >= time_before_onset)
        & (time_after_without >= time_after_onset)
    ]

    return final_onsets


def process_directory(
    videos_directory: Annotated[
        str,
        typer.Argument(help="Full path to the directory with videos and text files."),
    ],
    laser_color: Annotated[
        Optional[str],
        typer.Argument(
            help="Color of laser to extract, used only for bin laser files, 'blue' or 'orange'.",
            rich_help_panel="For .bin files",
            callback=laser_color_callback,
        ),
    ] = None,
    version: Annotated[
        Optional[bool],
        typer.Option("--version", callback=version_callback, is_eager=True),
    ] = None,
    time_before: Annotated[
        Optional[float],
        typer.Option(help="Time before the stimulation onset to include in clips."),
    ] = TIME_BEFORE_ONSET,
    time_after: Annotated[
        Optional[float],
        typer.Option(help="Time after the stimulation onset to include in clips."),
    ] = TIME_AFTER_ONSET,
    laser_min: Annotated[
        Optional[float],
        typer.Option(help="Minimum voltage to consider the laser 'on'."),
    ] = LASER_MIN_VOLTAGE,
    video_ext: Annotated[
        Optional[str], typer.Option(help="Video files extension.")
    ] = VIDEO_EXT,
    laser_ext: Annotated[
        Optional[str],
        typer.Option(help="Laser files extension (should be txt, csv or bin)."),
    ] = LASER_EXT,
    sep: Annotated[
        Optional[str], typer.Option(help="Separator used in text files.")
    ] = SEP,
    camera_min: Annotated[
        Optional[float],
        typer.Option(
            help="Minimum voltage to consider the camera laser 'on'. Used only for bin files."
        ),
    ] = CAMERA_MIN_VOLTAGE,
    sample_rate: Annotated[
        Optional[float],
        typer.Option(
            help="Recorded voltage traces sampling rate (set in Bonsai). Used only for bin files."
        ),
    ] = SAMPLE_RATE,
    adjust_factor: Annotated[
        Optional[float],
        typer.Option(help="Multiplicative correction factor. Used only for bin files."),
    ] = ADJUST_FACTOR,
):
    """
    Script to cut long recordings into small clips containing only one stimulation.
    
    """
    # prepare ffmpeg
    ffprobe_path, ffmpeg_path = get_ffmpeg_exe(FFMPEG_BIN_DIR)

    # list files to process
    files_list = find_videos_with_laser(
        videos_directory, videoext=video_ext, laserext=laser_ext
    )
    if len(files_list) < 2:
        print(f"Found {len(files_list)} video to process.")
    else:
        print(f"Found {len(files_list)} videos to process.")

    pbar = tqdm(files_list)
    for file in pbar:
        pbar.set_description(f"Processing {file}")
        # get file paths
        laser_path = os.path.join(videos_directory, file + "." + laser_ext)
        video_path = os.path.join(videos_directory, file + "." + video_ext)

        # prepare output directory for that video
        dirname = os.path.dirname(video_path)
        if laser_color:
            newdir = os.path.join(dirname, f"Cropped-{file}-{laser_color}")
        else:
            newdir = os.path.join(dirname, f"Cropped-{file}")
        if not os.path.isdir(newdir):
            os.mkdir(newdir)

        # read laser data
        if laser_path.endswith("txt") or laser_path.endswith("csv"):
            time, laser = read_txt_file(laser_path, sep=sep)
        elif laser_path.endswith("bin"):
            camera, laser = read_bin_file(laser_path, laser_color)
            time, laser = align_laser_video(
                camera, laser, camera_min, sample_rate, adjust_factor
            )
        else:
            ext = os.path.splitext(laser_path)[1]
            raise ValueError(
                f"Laser file extension '{ext}' not supported, use txt or bin"
            )

        # read video duration
        video_duration = find_video_duration(video_path, ffprobe_path)

        # find stimulation onsets
        onset_times = find_stim_onsets(time, laser, laser_min, TTHRESHOLD)
        # filter out the one that overlaps with another window
        onset_times = filter_stim_single_in_window(
            onset_times, time_before, time_after, video_duration
        )

        tqdm.write(f"Found {len(onset_times)} clips to extract.")
        pbar_sub = tqdm(onset_times)
        for idx, onset_time in enumerate(pbar_sub):
            pbar_sub.set_description(f"Writing {file}-{idx}")
            newvid_name = f"{file}-{idx}.mp4"
            newvid_path = os.path.join(newdir, newvid_name)
            if os.path.exists(newvid_path):
                tqdm.write(f"{newvid_name} already exists, skipping.")
                continue
            start = onset_time - time_before
            duration = time_before + time_after
            returncode = cut_video(
                ffmpeg_path, video_path, newvid_path, start, duration
            )
            if returncode != 0:
                raise SystemError(f"ffmpeg failed for {file}-{idx}.mp4")


# --- Call
if __name__ == "__main__":
    typer.run(process_directory)
