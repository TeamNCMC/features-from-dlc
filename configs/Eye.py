"""Configuration class.
This file is meant to be copied and edited to suit your needs. There are several
restrictions to its content, notably the Config class -- see corresponding docstrings.
Give it a sensible name, describing to which modality it corresponds (openfield, ...).

All global variables (in CAPSLOCK before the Class definition) should exist.
Remember to edit the `features_metrics_range` variable in the `get_features()` function
to adjust when the in-stim quantifying metric should be computed.

This particular version :
modality : eye
features : pupil angle
author : Guillaume Le Goc (g.legoc@posteo.org)
version : 2026.07.01
(This script is a slightly modified version of the 2024.11.20 version by Nella Znaor (znaor.nella@gmail.com))

"""

import os
import tomllib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

# --- Default values
# Those values are used for all animals if they are not found (in lowercase) in the
# settings.toml file, or if the latter does not exist.
PIXEL_SIZE = 2 / 56  # pixel size in mm
CLIP_DURATION = 0.2  # duration of the clip, this determines the frame-time conversion
STIM_TIME = (0.05, 0.1)  # (start, end) in same units as CLIP_DURATION
FRAMERATE = 400  # framerate for the common time vector, used to align all time series
STIM_DURATION = STIM_TIME[1] - STIM_TIME[0] # duration of the stimulation time

# --- Features parameters
# Make the stimulation onset be the time 0
SHIFT_TIME = True
# List of all bodyparts used in the DLC file
BODYPARTS = ["CR", "PupilRight", "PupilTop", "PupilLeft", "PupilBottom"]
# Features to normalize by subtracting their pre-stim mean. This must be a tuple, so if
# there is only one, write it like FEATURES_NORM = ("something",)
FEATURES_NORM = ("pupil_angle")
# Multiplier of standard deviation to define the initiation of reaction to determine the
# delay from stimulation onset
NSTD = 3
# Number of points to fit after signal is above NSTD times the pre-stim std
NPOINTS = 3
# Maximum allowed delay, above which it is not considered as a response
MAXDELAY = 0.05  # in seconds

MAX_RANGE = 3 # returns max val between (start_stim, start_stim + STIM_DURATION * MAXIMUM_RANGE) 
RANGE_END = [0.185, 0.195] 

# --- Data cleaning parameters
# Likelihood threshold, below which values will be interpolated.
LH_THRESH = 0.070
# If trace has more than this low-likelihood fraction of frames, drop it
LH_PERCENT = 0.5
# If trace has more than this low-likelihood consecutive frames, drop it
LH_CONSECUTIVE = 5
# Interpolating method to fill missing data, see doc for pandas.DataFrame.interpolate
# for options
INTERP_METHOD = "cubic"

# --- Display parameters
# X axis limits, empty list or None for automatic
XLIM = []
# X axis label for time series
XLABEL_LINE = "time (s)"
# Labels for each features, appears on the y axis of time series
FEATURES_LABELS = {"pupil_angle": "eye angle (deg)"}
# Preset y axes limits
FEATURES_YLIM = {"pupil_angle":[-25,25]}  # must be [ymin, ymax], empty {} for automatic
# Features to NOT plot
FEATURES_OFF = ()

# --- File-specific parameters
# Full path to the toml calibration file with the fit per animal
CALIBRATION_FILE = (
    r"E:\test_alexis\RightEye\Excitation\eye-calibration_GN198.toml"
)
# Eye to use, must match its name in the file above
EYEID = "Ipsi"


# --- Configuration class
class Config:
    """
    The configuration Class.

    Defines processing functions required by the features_from_dlc script, and reads the
    settings.toml file to provide pixel size and stimulation timings.

    The following methods must exist and return the proper variable types and dimensions
    to ensure the features_from_dlc script works as intended :
    - __init__()
    - read_setting()
    - get_pixel_size()
    - setup_time()
    - get_features()
    - write_parameters_file()
    - preprocess_df()
    Apart from those, any number of methods and attributes can be used to compute
    behavioral features and metrics.

    Parameters
    ----------
    settings_file : str
        Full path to the settings.toml file.
    animal : str or None, optional
        Animal ID present in the settings.toml file, or None to use default values.
        Here default refers to the [default] section of the settings.toml file, if any.
        Otherwise, defaults to the values defined above. Default is None.

    Returns
    -------
    cfg : Config object.

    """

    def __init__(self, settings_file: str | None = None):
        """Constructor."""
        # Read settings file
        if settings_file is None:
            settings_file = ""  # to work with os.path.exists
        if os.path.exists(settings_file):
            # use settings.toml file
            with open(settings_file, "rb") as fid:
                self.settings = tomllib.load(fid)
            self.clip_duration = self.read_setting("clip_duration", CLIP_DURATION)
            self.original_stim_time = self.read_setting("stim_time", STIM_TIME)
            self.framerate = self.read_setting("framerate", FRAMERATE)
        else:
            # Use global defaults
            warnings.warn(
                "No settings.toml file found, using global defaults from config file."
            )
            self.clip_duration = CLIP_DURATION
            self.original_stim_time = STIM_TIME  # before time shift
            self.framerate = FRAMERATE
            self.settings = None

        # Set animal ID
        self.animal = None

        # Define features computation
        (
            self.features,
            self.features_metrics,
            self.features_metrics_range,
            self.features_metrics_share,
            self.features_labels,
        ) = self.get_features()

        # Timings
        self.shift_time = SHIFT_TIME
        self.setup_time()  # will shift all times if requested

        # Features parameters
        self.bodyparts = BODYPARTS
        self.features_norm = FEATURES_NORM
        self.features_off = FEATURES_OFF
        self.maxrange = MAX_RANGE

        # Delay parameters
        self.nstd = NSTD
        self.npoints = NPOINTS
        self.maxdelay = MAXDELAY

        # Data cleaning parameters
        self.lh_thresh = LH_THRESH
        self.lh_percent = LH_PERCENT
        self.lh_consecutive = LH_CONSECUTIVE
        self.interp_method = INTERP_METHOD

        # Display parameters
        self.xlim = XLIM
        self.xlabel_line = XLABEL_LINE
        self.features_ylim = FEATURES_YLIM

    # ----------------------------------------------------------------------------------
    def read_setting(self, setting, fallback):
        """
        Read key from settings, with a fallback if not there.

        Parameters
        ----------
        setting : str
            Key.
        fallback : Any
            Default value to use if not found.

        Returns
        -------
        setting : value or fallback.
        """
        if setting in self.settings:
            return self.settings[setting]
        else:
            warnings.warn(
                f"A settings file was provided but {setting} could not be"
                f" read. Falling back to global default ({fallback})."
            )
            return fallback

    def setup_time(self):
        """
        Prepare time variables.

        Create the common time vector and shift all time variable so that stimulation
        onset is time 0. get_features() should be run before.

        """
        # common time vector for all time series
        self.time_common = np.linspace(
            0, self.clip_duration, int(self.clip_duration * self.framerate)
        )

        # shift stimulation times so that the onset is 0
        if self.shift_time:
            # time vector
            self.time_common = self.time_common - self.original_stim_time[0]

            # quantification metrics range
            features_metrics_range_original = self.features_metrics_range.copy()
            for key in features_metrics_range_original.keys():
                for key2 in features_metrics_range_original[key].keys():
                    self.features_metrics_range[key][key2] = [
                        val - self.original_stim_time[0]
                        for val in features_metrics_range_original[key][key2]
                    ]

            # stim timing
            self.stim_time = (
                0,
                self.original_stim_time[1] - self.original_stim_time[0],
            )
        else:
            self.stim_time = self.original_stim_time

    def get_pixel_size(self):
        """
        Parse pixel size from settings dictionary and animal.

        Pixel size is determined in this order and fallbacks to the next one.
        1. "pixel_size" key in the "animal" section of the settings.toml file.
        2. "pixel_size" key in the "default" section of the settings.toml file.
        3. PIXEL_SIZE global variable at the top of this file.
        This value is stored in `pixel_size` attribute.

        Parameters
        ----------
        settings : dict
            Parsed settings.toml file, or empty dict to use defaults defined here.

        """
        if not self.animal:
            self.pixel_size = PIXEL_SIZE
        elif self.settings:
            if self.animal in self.settings:
                self.pixel_size = self.settings[self.animal]["pixel_size"]
            elif "default" in self.settings:
                self.pixel_size = self.settings["default"]["pixel_size"]
            else:
                warnings.warn(
                    "A settings file was provided but the pixel size could not be"
                    f" read. Falling back to global default ({PIXEL_SIZE})."
                )
                self.pixel_size = PIXEL_SIZE
        else:
            self.pixel_size = PIXEL_SIZE

    def get_features(self) -> tuple[dict, dict, dict, dict, dict]:
        """
        Features and their definition.

        Must exist and return 5 dictionaries:
        - features : mapping a feature name to a lambda function. The latter must take a
        pandas.DataFrame as sole input and return a pandas.Serie or a 1D numpy array. It
        can call another method.
        - features_metrics : mapping a feature name to another dict. The latter maps the
        metric name and a lambda function that takes two arguments, the feature time
        serie and the corresponding time vector and returns a scalar. It quantifies the
        change of the feature during the stimulation.
        - features_metrics_range : same structure as `features_metrics`, but maps to
        list with two elements, defining the time range during which the metric is
        computed.
        - features_metrics_share : same structure as `features_metrics`, but maps to a
        bool, defining whether the metric should share its y-axis with the time serie,
        eg. the computed metric is in the same units as the feature itself.
        - features_labels : maps a feature to its displayed name on the y-axis of graph.

        """
        # How to compute features. It must be mapping between a feature name and a
        # lambda function that takes a DataFrame as a sole argument and returns a Serie
        # or a 1D numpy array. The function itself can call other methods defined in
        # this class file.
        features = {"pupil_angle": lambda df: -np.rad2deg(self.get_pupil_angle(df))}
        # How to compute the metric quantifying the change during stimulation. It must
        # be a mapping between a feature (defined above) and another dict. The latter
        # will map a name (that will be shown on top of the plot) to an actual
        # computation that returns a scalar (mean, max, etc.). The lambda function must
        # take 2 arguments, the first one being the time serie of the mapped feature and
        # the second one being the corresponding time vector (if time is not needed, use
        # `_`). Any number of metrics can be defined per feature.
        features_metrics = {"pupil_angle": {"mean": lambda val, _: np.mean(val), "end": lambda val, _: np.mean(val)}}

        # Select the time range in which the metric is computed, in the same units as
        # `stim_time`, before time-shifting is performed.
        features_metrics_range = {"pupil_angle": {"mean": [0.05, 0.1], "end":RANGE_END}}

        # Choose metrics that will have their y axis shared with the time series, eg.
        # when the metric is in the same units as the feature plotted. This is a similar
        # dict, with True and False.
        features_metrics_share = {"pupil_angle": {"mean": True, "end": True, "max": True, "delay": False}}

        # Labels for each features, appears on the y axis of time series
        features_labels = FEATURES_LABELS

        return (
            features,
            features_metrics,
            features_metrics_range,
            features_metrics_share,
            features_labels,
        )

    def write_parameters_file(
        self, outdir: str, name: str = "analysis_parameters.toml"
    ):
        """
        Saves (hardcoded) parameters used to analyze data and generate figures.

        Parameters
        ----------
        outdir : str
            Full path to output directory.
        name : str, optional
            File name. Default is "parameters.txt".

        """
        with open(os.path.join(outdir, name), "w") as fid:
            fid.writelines(f"date = {datetime.now().isoformat()}\n")
            fid.writelines(f"pixel_size = {self.pixel_size}\n")
            fid.writelines(f"stim_time = {list(self.stim_time)}\n")
            fid.writelines(f"clip_duration = {self.clip_duration}\n")
            fid.writelines(f"framerate = {self.framerate}\n")
            fid.writelines(f"nstd = {self.nstd}\n")
            fid.writelines(f"lh_thresh = {self.lh_thresh}\n")
            fid.writelines(f"lh_percent = {self.lh_percent}\n")
            fid.writelines(f"lh_consecutive = {self.lh_consecutive}\n")
            fid.writelines(f"interp_method = '{self.interp_method}'\n")

    def preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocessing function, marks data as missing.

        Must exist, take a pandas.DataFrame as sole input and return a DataFrame of the
        same size. If there's nothing to do, just return the input DataFrame.
        Otherwise, this is here you can mark data as missing based on custom criteria,
        that are not based on the likelihood. This exists because sometimes DLC is
        highly confident on a point that is badly placed, and there might be another way
        to find "bad" values.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            Same DataFrame, with np.nan where data should be considered as missing.

        """
        return df

    # ----------------------------------------------------------------------------------
    def get_pupil_angle(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convert pupil translational motion to angle movement.

        Eye angle is defined as :
        E = arcsin( (CR - P) / Rp)
        with CR : corneal reflection, P pupil center, Rp radius of rotation. The latter
        depends on the pupil diameter, so it has to be computed first, then the Rp vs
        pupil diameter calibration fit is used to get Rp.
        Pupil center and diameter are determined with 4 key points placed on top, left,
        bottom and right of the pupil.
        Fit is read from the `CALIBRATION_FILE` global variable.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with required keys.

        Returns
        -------
        pupil_angle : np.ndarray
            Eye angle time serie.

        """
        # mask values with low likelihood
        mask_pT = df["PupilTop"]["likelihood"] < 0.8 
        mask_pB = df["PupilBottom"]["likelihood"] < 0.8
        mask_pR = df["PupilRight"]["likelihood"] < 0.8
        mask_pL = df["PupilLeft"]["likelihood"] < 0.8

        n_low = mask_pT.astype(int) + mask_pB.astype(int) + mask_pR.astype(int) + mask_pL.astype(int)

        # get pupil center from all 4 pupil points or with only two
        pupil_centroid_x = df.loc[:, pd.IndexSlice[["PupilLeft", "PupilBottom", "PupilRight", "PupilTop"], "x"]].mean(axis=1)
        pupilL_pupilR_center_x = df.loc[:, pd.IndexSlice[["PupilLeft", "PupilRight"], "x"]].mean(axis=1)
        pupilT_pupilB_center_x = df.loc[:, pd.IndexSlice[["PupilTop", "PupilBottom"], "x"]].mean(axis=1)

        # Conditions
        cond_use_LR = (mask_pT | mask_pB) & ~mask_pR & ~mask_pL  
        cond_use_TB = (mask_pR | mask_pL) & ~mask_pT & ~mask_pB  
        cond_nan    = n_low >= 3 # ≥ 3 points mauvais → NaN

        # compute centroids and replace if conditions are met
        pupil_center_x = pupil_centroid_x.copy()
        pupil_center_x[cond_use_LR] = pupilL_pupilR_center_x[cond_use_LR] #remplace data if cond_use_LR is True
        pupil_center_x[cond_use_TB] = pupilT_pupilB_center_x[cond_use_TB]
        pupil_center_x[cond_nan] = np.nan

        # replace nan 
        pupil_center_x = (
            pupil_center_x.interpolate(limit=LH_CONSECUTIVE)
            .bfill(limit=LH_CONSECUTIVE)
            .ffill(limit=LH_CONSECUTIVE)
        )

        pupil_center_x = pupil_center_x.to_numpy()
    
        # get pupil diameter
        pupil_diameter = self.compute_diameter(
            df, "PupilLeft", "PupilRight", "PupilTop", "PupilBottom"
        )

        # get corresponding Rp
        with open(CALIBRATION_FILE, "rb") as fid:
            calib = tomllib.load(fid)
        a = calib[self.animal][EYEID]["a"]
        b = calib[self.animal][EYEID]["b"]
        Rp = b - a * pupil_diameter

        # compute angle
        pupil_angle = np.arcsin((df["CR", "x"] - pupil_center_x) / Rp)

        return pupil_angle

    def compute_distance(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """
        Compute the distance between two points for time series.
        `p1` and `p2` must be time series with frames on row, eg. shape (ntimes, 2).

        Parameters
        ----------
        p1, p2 : np.ndarray
            (ntimes, 2), (x, y) coordinates over time.

        Returns
        -------
        np.ndarray
            (ntimes, 1), distance between `p1` and `p2` over time.

        """

        return np.sqrt((p2[:, 0] - p1[:, 0]) ** 2 + (p2[:, 1] - p1[:, 1]) ** 2)

    def compute_diameter(
        self, df: pd.DataFrame, left: str, right: str, top: str, bottom
    ) -> np.ndarray:
        """
        Compute mean diameter of a circle from points on the left, right, top and bottom of
        the circle.

        Parameters
        ----------
        df : pd.DataFrame
            DLC DataFrame.
        left, right, top, bottom : str
            Keys in df.

        Returns
        -------
        np.ndarray
            Mean diameter time serie.

        """
        left_xy = df[left][["x", "y"]].to_numpy()
        right_xy = df[right][["x", "y"]].to_numpy()
        top_xy = df[top][["x", "y"]].to_numpy()
        bottom_xy = df[bottom][["x", "y"]].to_numpy()
        horiz = self.compute_distance(left_xy, right_xy)
        verti = self.compute_distance(top_xy, bottom_xy)
        return np.mean(np.array([horiz, verti]), axis=0)
