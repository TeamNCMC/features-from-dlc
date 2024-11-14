"""Configuration class.
This file is meant to be copied and edited to suit your needs. There are several
restrictions to its content, notably the Config class -- see corresponding docstrings.
Give it a sensible name, describing to which modality it corresponds (openfield, ...).

This particular version :
modality : openfield
features : speed, head angle, body angle, x, y
author : Guillaume Le Goc (g.legoc@posteo.org), Rémi Proville (Acquineuro)
version : 2024.11.14

"""

import os
import tomllib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from shapely import Polygon

# --- Default values
# Those values are used for all animals if they are not found (in lowercase) in the
# settings.toml file, or if the latter does not exist.
PIXEL_SIZE = np.mean([619 / 1935, 686 / 2190])  # pixel size in mm
CLIP_DURATION = 2  # duration of the clip, this determines the frame-time conversion
STIM_TIME = (0.5, 1)  # (start, end) in same units as CLIP_DURATION
FRAMERATE = 20  # framerate for the common time vector, used to align all time series

# --- Features parameters
# Make the stimulation onset be the time 0
SHIFT_TIME = True
# List of all bodyparts used in the DLC file
BODYPARTS = ["Left ear", "Right ear", "Tail", "Nose"]
# Features to normalize by subtracting their pre-stim mean. This must be a tuple, so if
# there is only one, write it like FEATURES_NORM = ("something",)
FEATURES_NORM = ("theta_body", "theta_neck")
# Multiplier of standard deviation to define the initiation of reaction to determine the
# delay from stimulation onset
NSTD = 3
# Number of points to fit after signal is above NSTD times the pre-stim std
NPOINTS = 3

# --- Data cleaning parameters
# Likelihood threshold, below which values will be interpolated.
LH_THRESH = 0.6
# If trace has more than this low-likelihood fraction of frames, drop it
LH_PERCENT = 0.5
# If trace has more than this low-likelihood consecutive frames, drop it
LH_CONSECUTIVE = 7
# Interpolating method to fill missing data, see doc for pandas.DataFrame.interpolate
# for options
INTERP_METHOD = "cubic"

# --- Display parameters
# X axis limits, empty list or None for automatic
XLIM = []
# X axis label for time series
XLABEL_LINE = "time (s)"
# Labels for each features, appears on the y axis of time series
FEATURES_LABELS = {
    "speed": "speed (cm/s)",
    "theta_body": "body angle (deg)",
    "theta_neck": "neck angle (deg)",
    "xbody": "centroid x (mm)",
    "ybody": "centroid y (mm)",
}
# Preset y axes limits
FEATURES_YLIM = {}  # must be [ymin, ymax], empty {} for automatic
# Features to NOT plot
FEATURES_OFF = ("xbody", "ybody")


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
        self.nstd = NSTD
        self.npoints = NPOINTS

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

    def get_features(self) -> tuple[dict, dict, dict, dict]:
        """
        Features and their definition.

        Must exist and return 4 dictionaries:
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
        features = {
            "speed": lambda df: self.get_speed(df, smooth=True) * self.pixel_size / 10,
            "theta_body": lambda df: np.rad2deg(
                self.get_theta_bodypart(
                    df, vec_0=("Tail", "neck"), reference="self", norm_win=(0, 10)
                )
            ),
            "theta_neck": lambda df: np.rad2deg(
                np.pi
                - self.get_theta_bodypart(
                    df, vec_0=("Nose", "neck"), reference=("Tail", "neck"), wrap=False
                )
            ),
            "xbody": lambda df: self.get_xy_com(df, "x") * self.pixel_size,
            "ybody": lambda df: self.get_xy_com(df, "y") * self.pixel_size,
        }
        # How to compute the metric quantifying the change during stimulation. It must
        # be a mapping between a feature (defined above) and another dict. The latter
        # will map a name (that will be shown on top of the plot) to an actual
        # computation that returns a scalar (mean, max, etc.). The lambda function must
        # take 2 arguments, the first one being the time serie of the mapped feature and
        # the second one being the corresponding time vector (if time is not needed, use
        # `_`). Any number of metrics can be defined per feature.
        features_metrics = {
            "speed": {
                "mean": lambda val, _: np.mean(val),
                "deceleration": lambda s, t: -self.get_accel_coef(s, t),
            },
            "theta_body": {"max": lambda val, _: np.max(val)},
            "theta_neck": {"max": lambda val, _: np.max(val)},
            "xbody": {},
            "ybody": {},
        }

        # Select the time range in which the metric is computed, in the same units as
        # `stim_time`, before time-shifting is performed.
        features_metrics_range = {
            "speed": {"mean": [0.5, 1], "deceleration": [0.5, 0.75]},
            "theta_body": {"max": [0.5, 1]},
            "theta_neck": {"max": [0.5, 1]},
            "xbody": {},
            "ybody": {},
        }

        # Choose metrics that will have their y axis shared with the time series, eg.
        # when the metric is in the same units as the feature plotted. This is a similar
        # dict, with True and False.
        features_metrics_share = {
            "speed": {"mean": True, "deceleration": False},
            "theta_body": {"max": True},
            "theta_neck": {"max": True},
            "xbody": {},
            "ybody": {},
        }

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
        df_in = df.copy()

        # drop likelihood
        df_proc = df_in.drop("likelihood", level=1, axis=1)

        # loop through each frames
        for irow, row in df_proc.iterrows():
            # get x,y of polygon that defines the body
            pts = (
                row[["Left ear", "Nose", "Right ear", "Tail", "Left ear"]]
                .to_numpy()
                .reshape((5, 2))
            )
            mouse = Polygon(pts)
            if not mouse.is_simple:
                # if the polygon is not closed, that's because the Nose bodypart was not
                # placed correctly so we consider it as missing.
                df_in.loc[irow, ("Nose", "x")] = np.nan
                df_in.loc[irow, ("Nose", "y")] = np.nan

        return df_in

    # ----------------------------------------------------------------------------------

    # Methods below are user-defined and can be used to derive features from bodyparts
    def get_speed(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Wraps center of mass speed time serie computation.

        `df` must have the following columns :
        ["Left ear", "Nose", "Right ear", "Tail", "Left ear"], with "x" and "y" for each.
        `fps` is computed here, do not provide it in the **kwargs.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with required keys.
        **kwargs : passed to `speed_bodypart`.

        Returns
        -------
        speed : np.ndarray
            Speed of center of mass in pixels/time. Units depend on `clip_duration`.

        """
        df_dlc = df.copy()
        df_dlc = self.compute_center_of_mass(df_dlc)  # add "centroid" column
        fps = len(df_dlc) / self.clip_duration

        return self.speed_bodypart(df_dlc, "centroid", fps, **kwargs)

    def get_theta_bodypart(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Wraps relative angle time serie computation.

        `df` must have the following columns :
        ["Left ear", "Nose", "Right ear"], with "x" and "y" for each.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with required keys.
        **kwargs : passed to `get_angle`.

        Returns
        -------
        theta : np.ndarray
            Time series of requested angle in radians.

        """
        df_dlc = df.copy()
        df_dlc = self.compute_mid_head(df_dlc)  # add "neck" column

        return self.get_angle(df_dlc, **kwargs)

    def compute_mid_head(self, df_dlc: pd.DataFrame):
        """
        Given a dataframe of DLC tracked points, computes the middle of the head and a point
        on the neck. Those points are then named mid_head and neck.

        mid_head: it is the barycenter of Left ear, Nose and Right ear.
        neck: it is the intersection point between a line passing through the ears and a
        line passing through the nose and the mid_head point.

        From Aquineuro.

        Parameters
        ----------
        df_dlc: pd.DataFrame
            Dataframe containing the coordinates of tracked bodyparts

        Returns
        -------
        df_dlc: pd.DataFrame
            Dataframe containing the coordinates of tracked bodyparts
            with added columns named 'mid_head' and 'neck'
        """
        df = df_dlc.copy()
        df = df.drop("likelihood", level=1, axis=1)
        head = df.loc(1)[["Left ear", "Nose", "Right ear"]]
        # modified by GLG to work with pandas>=2.1
        coords = (
            head.melt(ignore_index=False)
            .reset_index()
            .rename(columns={"index": "frame"})
        )
        mid_head = coords.groupby(["frame", "coords"]).mean(["x", "y"])
        df_dlc[("mid_head", "x")] = mid_head.loc(axis=0)[:, "x"].reset_index()["value"]
        df_dlc[("mid_head", "y")] = mid_head.loc(axis=0)[:, "y"].reset_index()["value"]
        df = df_dlc.copy()
        df = df.drop("likelihood", level=1, axis=1)
        ears = df[["Left ear", "Right ear"]].to_numpy().reshape((-1, 2, 2))
        head = df[["Nose", "mid_head"]].to_numpy().reshape((-1, 2, 2))
        neck = np.array(
            [self.find_intersection(c_ear, c_head) for c_ear, c_head in zip(ears, head)]
        )
        df_dlc[("neck", "x")] = neck[:, 0]
        df_dlc[("neck", "y")] = neck[:, 1]

        return df_dlc

    def compute_center_of_mass(self, df_dlc: pd.DataFrame):
        """
        Given a dataframe of DLC tracked points, computes the center of mass.

        In order to estimate the animal shape we compute a polygon defined by
        ['Left ear', 'Nose', 'Right ear', 'Tail', 'Left ear'].
        Left ear is there twice to close the polygon. Centroid computation is done by
        the shapely package.

        From Aquineuro.

        Parameters
        ----------
        df_dlc: pd.DataFrame
            Dataframe containing the coordinates of tracked bodyparts

        Returns
        -------
        df_dlc: pd.DataFrame
            Dataframe containing the coordinates of tracked bodyparts with an added column
            named 'centroid'.

        """
        dlc_dl = df_dlc.drop("likelihood", level=1, axis=1)
        df_dlc[("centroid", "x")] = np.nan
        df_dlc[("centroid", "y")] = np.nan

        for irow, row in dlc_dl.iterrows():
            pts = (
                row[["Left ear", "Nose", "Right ear", "Tail", "Left ear"]]
                .to_numpy()
                .reshape((5, 2))
            )

            mouse = Polygon(pts)
            c_of_m = mouse.centroid

            df_dlc.loc[irow, ("centroid", "x")] = c_of_m.x
            df_dlc.loc[irow, ("centroid", "y")] = c_of_m.y

        return df_dlc

    def get_xy_com(self, df_dlc: pd.DataFrame, axis: str) -> np.ndarray:
        """
        Returns centroid x or y coordinates as numpy array.

        Parameters
        ----------
        df_dlc : pd.DataFrame
            Dataframe containing the coordinates of tracked bodyparts
        axis : {"x", "y"}
            x or y coordinates.

        Returns
        -------
        coord : np.ndarray
            x or y coordinates time serie of the center of mass.

        """
        df_in = df_dlc.copy()
        df_com = self.compute_center_of_mass(df_in)
        return df_com[("centroid", axis)].to_numpy()

    def speed_bodypart(
        self, df_dlc: pd.DataFrame, bodypart: str, fps, smooth=False, window_dur=0.5
    ) -> np.ndarray:
        """
        Compute the speed of a given bodypart from the DLC dataframe.

        Uses the `compute_speed` function. See there for details.

        From Aquineuro.

        Parameters:
        ----------
        df_dlc : pd.DataFrame
            Dataframe containing the coordinates of tracked bodyparts
        bodypart : str
            Bodypart to use for velocity computation.
        fps : int
            Frame rate at which the coordinates are acquired. Most likely the video
            framerate.
        smooth : bool
            Should we use a Savitzky-Golay filter ? Default is False.
        window_dur : float
            If Savitzky-Golay filter is used, this is the window it uses. Unit: second.

        Returns
        -------
        velocity: np.ndarray
            Computed velocity.

        """
        coords = df_dlc[bodypart][["x", "y"]].to_numpy().T
        if smooth:
            deg = 3
        else:
            deg = None
        velocity = self.compute_speed(coords, window_dur, fps, deg)
        return velocity

    def compute_speed(
        self, mouse_xy: np.ndarray, window_dur: float, fps: int, deg: int = 3
    ) -> np.ndarray:
        """
        Given an array of coordinates, compute the speed.

        There is the option of filtering the coordinates first using a Savitzky-Golay
        filter, this option is deactivated if the given degree is None. Velocity is computed
        as the norm of the vector made from the gradients of the x and y vectors.

        From Aquineuro.

        Parameters
        ----------
        mouse_xy : np.ndarray
            Array of coordinates.
            First row is the x and second row the y. Each colums is a different time point.
        window_dur : float
            If Savitzky-Golay filter is used, this is the window it uses. Unit: second.
        fps : int
            Frame rate at which the coordinates are acquired. Most likely the video
            framerate.
        deg : int, optional
            Degree of the polynoms used by the Savitzky-Golay filter. If None, filtering is
            disabled (default).

        Returns
        -------
        speed: np.ndarray
            Same length as `mouse_xy`, but 1D.

        """
        w = window_dur * fps
        w = int((w // 2) * 2 + 1)

        xm, ym = mouse_xy[0, :], mouse_xy[1, :]
        if deg is not None:
            smooth_x = savgol_filter(xm, w, deg)
            smooth_y = savgol_filter(ym, w, deg)
        else:
            smooth_x = xm.copy()
            smooth_y = ym.copy()
        # Time in seconds
        t = np.arange(len(xm)) / fps
        # Gradients for speed
        dx = np.gradient(smooth_x, t)
        dy = np.gradient(smooth_y, t)
        velocity = np.vstack((dx, dy))
        speed = np.linalg.norm(velocity, axis=0)

        return speed

    def find_intersection(self, line_0, line_1):
        """
        Compute the intersection point of two lines and returns its coordinates.

        From Aquineuro.

        Parameters
        ----------
        line_0 : np.ndarray
            2x2 array
            [[x1, y1],
            [x2, y2]]
        line_1 : np.ndarray
            2x2 array
            [[x3, y3],
            [x4, y4]]

        Returns
        -------
        x : float
        y : float

        """
        stacked_lines = np.vstack((line_0, line_1))
        x1, x2, x3, x4 = stacked_lines[:, 0]
        y1, y2, y3, y4 = stacked_lines[:, 1]
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return
        inter_x = (
            (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        ) / denominator
        inter_y = (
            (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        ) / denominator
        return inter_x, inter_y

    def get_angle(
        self,
        df_dlc,
        vec_0=("Nose", "mid_head"),
        reference=None,
        baseline_len=10,
        wrap=True,
        norm_win=(0, 10),
    ):
        """
        Computes the angle time serie between two lines.

        The two lines are defined by the two bodyparts in `vec_0` and `reference`.

        From Aquineuro.

        Parameters
        ----------
        df_dlc : pd.DataFrame
            Dataframe containing the coordinates of tracked bodyparts.
        vec_0 : tuple of str
            Name of two tracked points used to form the vector of which we'll compute the
            angle.
        reference: str or tuple of str, optional
            Reference vector. Can be:
                None : Compute the angle relative to the horizontal (default).
                'self' : Compute the angle relative to the vector that is the average of the
                        vector of interest over the first `baseline_len` frames
                tuple of str : Name of two tracked points to form a vector relative to which
                            the angle is computed.
        baseline_len : int
            Number of frames over wich we average the vector to serve as a reference.
        wrap: bool, optional
            Should the angles we phase-wrapped and normalized ? Default is True.
        norm_win: tuple, optional
            Time window (in index) to use to normalize the angles. Default is (0, 10).

        Returns
        -------
        theta: np.array
            Computed angle, over time, in radian.

        """
        u_vec = self.get_norm_vector(df_dlc, *vec_0)
        if reference is None:
            # Horizontal vector
            ref_vec = np.tile(np.array([1, 0]), (u_vec.shape[1], 1)).T
        elif reference == "self":
            # Average vector
            self_init_vec = u_vec[:, :baseline_len].mean(1)
            ref_vec = np.tile(self_init_vec, (u_vec.shape[1], 1)).T
        else:
            # Reference vector from two other points
            ref_vec = self.get_norm_vector(df_dlc, *reference)
        # cos from scalar product, sin from cross product
        cos_theta = u_vec[0] * ref_vec[0] + u_vec[1] * ref_vec[1]
        sin_theta = np.cross(u_vec, ref_vec, axis=0)
        # Todo: Check on more cases
        # Get the angle from the arccos
        theta = np.arccos(np.clip(cos_theta, -1, 1))
        if np.any(np.isnan(theta)):
            theta[np.isclose(cos_theta, 1)] = 0
        # Set theta in the proper half circle depending on the sign of the sine
        if wrap:
            theta[sin_theta < 0] = 2 * np.pi - theta[sin_theta < 0]
            theta = self.unwrap_norm_angle(theta, norm_win)

        return theta

    def get_norm_vector(self, df_dlc, point_0="Nose", point_1="mid_head"):
        """
        From two points on the animal, build a normalized (norm=1) vector for each time
        point.

        From Aquineuro.

        Parameters
        ----------
        df_dlc: pd.DataFrame
            Dataframe containing the coordinates of tracked bodyparts
        point_0: str
            Name of the first body part, defining the vector. Must be a column in the dataframe
            Default is 'Nose'
        point_1: str
            Name of the second body part, defining the vector. Must be a column in the dataframe
            Default is 'mid_head'

        Returns
        -------
        u_vec: np.ndarray
            2 rows (x, y) and as many columns as there are time points.
            This is the normalized vector for each time point
            Trnasposed compared to the dataframe, because easier to use
        """
        c0 = df_dlc[point_0][["x", "y"]].to_numpy()
        c1 = df_dlc[point_1][["x", "y"]].to_numpy()
        vec = c0 - c1
        u_vec = vec.T / np.linalg.norm(vec, axis=1)  # Careful, transposed, 2 rows now
        return u_vec

    def unwrap_norm_angle(self, angle, norm_win):
        """
        Handle the phase issue one can have with trigonometry functions.

        Ensure there are no large delta in the angle vector by using the numpy.unwrap
        function. Also, normalize it.

        From Aquineuro.

        Parameters
        ----------
        angle: np.ndarray
            Vector

        Returns
        -------
        angle: np.ndarray
            Unwrapped and normalized angles
        norm_win: tuple
            Time window (in index) to use to normalize the angles.

        """
        angle = np.unwrap(angle)
        if norm_win is not None:
            bsl = angle[slice(*norm_win)].mean()
        else:
            bsl = 0
        angle -= bsl
        return angle

    def get_accel_coef(
        self, speed: np.ndarray | pd.Series, time: np.ndarray | pd.Series
    ) -> float:
        """
        Compute the acceleration coeficient from the speed time serie.

        The acceleration coefficient is approximated by the mean derivative in the selected
        time window.

        Parameters
        ----------
        speed : np.ndarray or pd.Series
            (ntimes, 1) time serie.
        time : np.ndarray or pd.Series
            (ntimes, 1), corresponding time vector.

        Returns
        -------
        a : float or pd.Series
            Acceleration coefficient.

        """
        return np.mean(np.gradient(speed, time))
