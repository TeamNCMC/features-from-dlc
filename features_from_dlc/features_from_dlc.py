"""
Package to display features computed from body parts tracked with DeepLabCut (DLC).

The entry point of the package is the `process_directory()` function.

The files generated with DLC, either csv or h5, are supported. They should all be in the
same directory.
Their file names is important : it is used to recognize the animals identity and
conditions. A file must :
- contain one and only one trial
- begin with the animal ID
- contain a bit that uniquely identifies a condition

Next to the files to be analyzed, a settings.toml file can be copied from the resources
folder. It specifies the physical parameters of the experiment : video duration,
stimulation timings and pixel size. The latter can be specified per-animal.
If this file does not exist or there are missing information in it, the default values
from the configuration file are used.

This module loads those files, computes features as defined in the configuration module,
averages them across groups, computes metrics that quantify the change of the feature
during the stimulation and plot all that, along with a raster plot of all trials.
Additionnaly, the delay from the stimulation onset and the behavioral response is
estimated.

Statistical significance tests are performed. Non-parametrics tests are used. First, an
overall test is performed across conditions. If a significant difference is found
between groups, post-hoc pairwise tests are performed and stars are plotted accordingly
between pairs of consecutive conditions. The full significance table for each metrics
can also be saved to examine all pairs.
For paired data : Friedman test (or Cochran for binary data), then pairwise Wilcoxon
tests.
For unpaired data : Kruskal-Wallis test, then pairwise Mann-Whitney U tests.

It can save :
- a CSV file with each trials' time series and metadata (animal, condition...),
- a CSV file with in-stim metrics,
- a CSV file with response rate and delays,
- CSV files for each metrics and delays with pairwise significance tests,
- figures (svg),
- log files with files used for analysis and files dropped because there were too
much tracking errors (low likelihood),
- a summary of the analysis parameters that were used, for reference.

Information
-----------
- Physical parameters (pixel size, clip duration and stimulation timings are specified
in a settings.toml file next to the files to be analyzed. Otherwise, default values from
the configuration file will be used.
- Modality-specific parameters are specified in configs/xxx.py files.
- Plot style is defined in a separate config_plot.toml file. See there for options.
- If an output directory is defined (`outdir`), existing files will be
overridden.
- You can write your own configuration file. Just copy one that is working and modify
the computations to your need, respecting the requirements to be compatible with this
script.

If you're lost, check out the README file, and/or drop me a line :)

"""

# --- Imports
import importlib
import os
import tomllib
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from matplotlib import rcParams
from rich import print
from tqdm import tqdm

pd.options.mode.copy_on_write = True  # prepare for pandas 3.0


def get_config(modality: str, path_to_configs: str, settings_file: str | None = None):
    """
    Import configuration module.

    A `modality`.py file defining the Config class must exist in the `path_to_configs`
    directory.

    Parameters
    ----------
    modality : str
        Name of the modality
    path_to_configs : str
        Full path to where are the configuration files.
    settings_file : str, optional
        Full path to the optional settings.toml file.

    Returns
    -------
    cfg : Config

    """
    module_path = os.path.join(path_to_configs, modality + ".py")
    try:
        spec = importlib.util.spec_from_file_location(modality, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError:
        err_msg = f"Could not find '{modality}.py in the '{path_to_configs}' directory."
        raise ImportError(err_msg)

    return module.Config(settings_file)


def setup_plot_style(
    style_file: str, plot_animal_monochrome: bool = False, nanimals: int = 0
):
    """
    Setup plots styles defined by the TOML configuration file.

    The configuration file is read and matplotlib's `rcParam` is set accordingly.
    The returned dictionary has keys "stim", "pooled", "sem", "animal" and "trial",
    the latters containing the arguments to pass as **kwargs to corresponding plotting
    functions.
    It also has keys "figsize", "arrows", "nxticks" and "nyticks".

    Parameters
    ----------
    style_file : str
        Full path to the TOML configuration file.
    plot_animal_monochrome : bool, optional
        Whether to plot different animals in the same color, default is False.
    nanimals : int, optional
        Number of animals, required only if `plot_animal_monochrome` is True.

    Returns
    -------
    kwargs_plots : dict
        Arguments to pass to various plotting functions.

    """
    with open(style_file, "rb") as fid:
        config = tomllib.load(fid)  # read config file

    # Setup matplotlib
    new_rc_params = {
        "svg.fonttype": "none",  # store text as text
        "font.family": config["axes"]["fontfamily"],
        "font.size": config["axes"]["fontsize"],
        "axes.linewidth": config["axes"]["linewidth"],
        "xtick.major.width": config["axes"]["linewidth"],
        "ytick.major.width": config["axes"]["linewidth"],
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.grid": config["axes"]["grid"],
    }
    rcParams.update(new_rc_params)
    plt.ion()  # interactive figures

    # Parse options
    if plot_animal_monochrome:
        config_animal = config["animal_monochrome"]
        config_animal["color"] = nanimals * [config_animal["color"]]
    else:
        config_animal = config["animal_color"]

    # Return plotting functions arguments
    return {
        "stim": config["stim"],
        "pooled": config["pooled"],
        "sem": config["sem"],
        "animal": config_animal,
        "condition": config["condition"],
        "trial": config["trial"],
        "metrics": config["metrics"],
        "delays": config["delays"],
        "figsize": config["axes"]["figsize"],
        "arrow": config["axes"]["arrow_tip"],
        "arrow_size": config["axes"]["arrow_size"],
        "nxticks": config["axes"]["nxticks"],
        "nyticks": config["axes"]["nyticks"],
    }


def get_condition(filename: str, conditions_map: dict) -> str | None:
    """
    Get condition name based on file name.

    `conditions` maps a condition name (eg. "control", "10mW", "20mW") to a filter
    for the file name. If any of the strings in the filter is contained in the file
    name, then the corresponding condition is returned as string.
    The beginning of the file name has priority : if a file name begins with a filter,
    the corresponding condition is returned, whether this file name contains other
    filters (see example below).

    Example :
        conditions = {"control": ["animal70"], "low pow.": ["10mW"], "high pow.": ["20mW"]}
        filename -> returned condition with this function :
            - animal70_10mW_blabla.h5 -> "control"
            - animal70_20mW_blabla.h5 -> "control"
            - animal71_10mW_blabla.h5 -> "low pow."
            - animal81_20mW_blabla.h5 -> "high pow."

    Parameters
    ----------
    filename : str
        File name.
    conditions_map : dict
        In the form {name: [filters]}.

    Returns
    -------
    name : str or None
        Name of the condition, or None if it is not found in the file name.

    """
    condition = None

    for key, values in conditions_map.items():  # loop through defined groups
        # convert to tuple
        if isinstance(values, str):
            values = (values,)
        elif isinstance(values, list):
            values = tuple(values)

        # first, check if the file starts with the value, this has priority
        if filename.startswith(values):
            return key

        # then check if the file contains any of the other strings
        if np.any([v in filename for v in values]):
            if not condition:
                # condition is still None : it was not attributed to a group yet
                condition = key
            else:
                # condition was already attributed, so most likely the same filter was
                # used in several groups, which is not allowed.
                raise ValueError(
                    (
                        "'CONDITIONS' is not valid. Most likely, the same filter was "
                        "used in several groups."
                    )
                )

    return condition


def convert_frames_to_time(frames, duration):
    """
    Convert frames to time based on the clip duration.

    Parameters
    ----------
    frames : array-like
        Vector of frames.
    duration : float
        Duration of the clip.

    Returns
    -------
    time : array-like
        Vector of time, in units of `duration`.

    """
    return ((frames - frames.min()) * duration) / frames.max()


def clean_df(
    df: pd.DataFrame,
    likelihood_thresh: float,
    likelihood_percent: float,
    likelihood_consecutive: int,
    bodyparts: list = [],
    custom_filter_fun=None,
    **kwargs,
) -> pd.DataFrame:
    """
    Interpolate values where likelihood is low.

    `df` must be a MultiIndex with two levels, on the second level there must be
    "x", "y" and "likelihood" columns, eg. the DataFrame must come from DeepLabCut.
    First, the preprocessing function from the configuration file is exectued on the
    DataFrame, marking missing data based on custom criteria (see
    `configs.preprocess_df`). Then, it filters out low-likelihood data :
    Time steps where the likelihood is below `likelihood_thresh` will be replaced by NaN
    then it is interpolated. The time serie will be considered unusable and dropped if
    any of those conditions are met :
        - it has more than `likelihood_percent` (in fraction) time steps with low
        likelihood OR
        - it has more than `likelihood_consecutive` consecutive time steps with low
        likelihood

    Parameters
    ----------
    df : pandas.DataFrame
    likelihood_thresh : float
        Consider values below this as missing and interpolate it.
    likelihood_percent : float
        If more than this fraction of the trace is missing, drop the trial.
    likelihood_consecutive : int
        If more than this number of consecutive frames of the trace is missing, drop the
        trial.
    bodyparts : list, optional
        Bodyparts to use, everything else is removed. Default is empty list.
    custom_filter_fun : function, optional
        Apply a custom function to mark data as np.nan (missing). This function should
        take as input the DataFrame and return the same DataFrame with np.nan where the
        data is missing.
    **kwargs : passed to pandas.DataFrame.interpolate()
        `limit` keyword argument is set to `likelihood_consecutive`.

    Returns
    -------
    df : pandas.DataFrame
        Same `df` as input, with replaced values. If the trial is to be dropped,
        returns empty DataFrame.

    """
    if len(bodyparts) == 0:
        df_in = df.copy()
    else:
        df_in = df.copy()[bodyparts]

    # apply custom marking-as-missing function
    if custom_filter_fun is not None:
        df_in = custom_filter_fun(df)

    # get columns with a "likelihood" sub-column
    cols = df_in.columns[df_in.columns.get_level_values(1).isin(["likelihood"])]

    for col in cols:
        # replace values with nan
        df_in.loc[df_in[col] < likelihood_thresh, (col[0], ["x", "y"])] = np.nan

        # check if we keep this trial
        if df_in.loc[:, (col[0], "x")].isna().sum() / len(df_in) > likelihood_percent:
            return pd.DataFrame()

    # fill nan with interpolation
    df_in = (
        df_in.interpolate(limit=likelihood_consecutive, **kwargs)
        .bfill(limit=likelihood_consecutive)
        .ffill(limit=likelihood_consecutive)
    )

    # check if it was filled, if not, it means we drop this trial
    if df_in.isnull().values.any():
        return pd.DataFrame()

    return df_in


def read_dlc_file(filename: str, header: None | int | list = [1, 2]) -> pd.DataFrame:
    """
    Reads file output from DeepLabCut.

    Wraps pandas.read_csv() or pandas.read_hdf() depending on `filename` extension.

    Parameters
    ----------
    filename : str
        Full path to the DeepLabCut file.
    header : None, int or list, optional
        Passed to pandas.read_csv(). Default is [1, 2] (to read DeepLabCut CSV files).

    Returns
    -------
    df : pandas.DataFrame

    """
    if filename.endswith(".csv"):
        df = pd.read_csv(filename, sep=",", skiprows=0, index_col=0, header=header)
    elif filename.endswith(".h5"):
        df = pd.read_hdf(filename)  # read table
        df = df[df.columns.get_level_values(0)[0]]  # remove "scorer" level

    return df


def write_trial_file(trial_name: str, filename: str | None):
    """
    Write `trial_name` to `filename`.

    Parameters
    ----------
    trial_name : str
        Trial identifier (eg. file name).
    filename : str or None
        Path to the file where it is written. If it's None, no writing is done.

    """
    if filename:
        with open(filename, "a") as fid:
            fid.writelines(trial_name + "\n")


def process_animal(
    files_list: list,
    animal: str,
    conditions: dict,
    cfg,
    dropfile: str | None = None,
    usefile: str | None = None,
) -> pd.DataFrame:
    """
    Finds CSV files starting with specified animals ID, converts them to DataFrame.

    Extracts the coordinates we're interested in, compute the feature we want and
    align them on a single time vector. Returns the data labelled with animal and
    trial number.

    Parameters
    ----------
    files_list : list
        List of CSV or H5 files.
    animal : str
        Animal ID.
    conditions : dict
        Conditions names and identifiers.
    cfg : Config
        Config object from configuration file.
    dropfile, usefile : str or None, optional
        Files that will log dropped and used trials. Default is None (no file written).

    Returns
    -------
    df_animal : pandas.DataFrame
        DataFrame with animal-labelled data.

    """
    # limit files list to the animal
    files_animal = [
        file for file in files_list if os.path.basename(file).startswith(animal)
    ]

    # check there are files
    if len(files_animal) == 0:
        raise FileNotFoundError(f"No file found for animal '{animal}'.")

    # set animal id and update pixel size
    cfg.animal = animal
    cfg.get_pixel_size()

    df_animal_list = []  # prepare the aligned DataFrame list
    trial = 0  # initialize trial counter

    pbar = tqdm(files_animal)
    for file in pbar:  # loop through files
        pbar.set_description(f"Processing trial {trial}")

        basename = os.path.basename(file)

        # read data (with multi-indexed columns)
        df_in = read_dlc_file(file, header=[1, 2])

        # data cleaning
        df_in = clean_df(
            df_in,
            cfg.lh_thresh,
            cfg.lh_percent,
            cfg.lh_consecutive,
            bodyparts=cfg.bodyparts,
            custom_filter_fun=cfg.preprocess_df,
            method=cfg.interp_method,
        )

        # check if we drop this trial because too much low-likelihood values
        if df_in.empty:
            pbar.write(f"[Info] {basename} dropped.")
            write_trial_file(basename, dropfile)
            continue

        # parse condition based on file name
        condition = get_condition(basename, conditions)
        if not condition:
            # the condition was not parsed from the file name, so we drop it
            pbar.write(
                f"[Warning] Not able to attribute a condition to {basename},"
                " skipping."
            )
            continue

        df_tmp = pd.DataFrame()  # initialize DataFrame where we collect relevant data
        df_out = pd.DataFrame()  # initialize output DataFrame

        # get frames number
        df_tmp["frames"] = df_in.index

        # convert to frames to time
        df_tmp["time"] = convert_frames_to_time(df_tmp["frames"], cfg.clip_duration)
        if cfg.shift_time:
            # shift time with the original, unshifted, stim timings
            df_tmp["time"] = df_tmp["time"] - cfg.original_stim_time[0]

        df_out["time"] = cfg.time_common  # common time vector for all time series

        # compute requested features
        for feature, computation in cfg.features.items():
            # compute feature
            df_tmp[feature] = computation(df_in)

            # normalize if requested
            if feature in cfg.features_norm:
                df_tmp[feature] = (
                    df_tmp[feature]
                    - df_tmp[feature][df_tmp["time"] < cfg.stim_time[0]].mean()
                )

            # interpolate data on the common time vector and add it to the new DataFrame
            # we do this to ensure that all data are sampled on the same time vector.
            df_out[feature] = np.interp(df_out["time"], df_tmp["time"], df_tmp[feature])

        # add trial number
        df_out["trial"] = trial

        # add unique trial ID
        df_out["trialID"] = f"{animal}-{trial}"

        # add condition
        df_out["condition"] = condition

        # keep track of the file name
        df_out["filename"] = basename

        # add new data to the rest
        df_animal_list.append(df_out)

        # increment the trial number
        trial += 1

        # log the file
        write_trial_file(basename, usefile)

    # concatenate all DataFrames
    if len(df_animal_list) == 0:
        raise ValueError(
            (
                f"No data for {animal}. Most likely all trials were dropped or "
                "files were not assigned any condition."
            )
        )
    df_animal = pd.concat(df_animal_list)

    # label the data with the animal ID
    df_animal["animal"] = animal

    return df_animal


def get_pvalue_timeserie(
    df: pd.DataFrame, key: str, stim_time: list | tuple
) -> float | None:
    """
    Perform a paired ttest between mean values before stim and after stim.

    All trials are pooled, so the test will only be performed if only one condition is
    present.

    Parameters
    ----------
    df : pandas.DataFrame
    y : str
        Key in `df`.
    stim_time : 2-elements tuple or list
        Stimulation onset and offset.

    Returns
    -------
    pvalue : float or None
        p-value or None if no test was performed.

    """
    # check only one condition
    if df["condition"].nunique() > 1:
        return None

    # pre-stim
    dfpre = df[df["time"] < stim_time[0]]
    dfpre_group = dfpre.groupby("trialID")  # extract time series
    mean_prestim = dfpre_group[key].mean().values  # get mean values

    # during-stim
    dfin = df[(df["time"] >= stim_time[0]) & (df["time"] <= stim_time[1])]
    dfin_group = dfin.groupby("trialID")  # extract time series
    mean_instim = dfin_group[key].mean().values  # get mean values

    # paired ttest
    result = pg.ttest(mean_prestim, mean_instim, paired=True)

    return result.loc["T-test", "p-val"]


def perform_stat_test(
    df: pd.DataFrame,
    measurement_names: list,
    paired: bool = False,
    pthresh: float = 0.05,
) -> dict[pd.DataFrame]:
    """
    Perform statistical tests.

    Input `df` is expected to have columns "condition" and "animal", in addition of
    measurement names contained in `measurement_names`.
    Non-parametric pairwise tests are performed between each condition. If paired,
    Mann-Whitney is used, otherwise, Wilcoxon signed rank is used. For more information,
    see (1).

    A list of DataFrame the same size as `measurement_names` is returned, containing all
    information about the result of the pairwise tests.

    (1) https://pingouin-stats.org/build/html/generated/pingouin.pairwise_tests.html

    Parameters
    ----------
    df : pd.DataFrame
        Measurements DataFrame.
    measurements_names : list
        List of measurement names in `df`.
    paired : bool, optional
        Whether to perform paired tests (repeted measurements). Default is False.
    pthresh : float, optional
        Threshold of p-value of the overall test before post-hoc pairwise tests. Default
        is 0.05.

    Returns
    -------
    results : dict of DataFrame
        Map a measurement name to a DataFrame with test result summary as returned by
        `pingouin.pairwise_tests()`.

    """
    results = {}
    ncondtions = df["condition"].nunique()

    for meas_name in measurement_names:
        # check if binary values
        if df[meas_name].nunique() == 2:
            binary = True
        else:
            binary = False
        if ncondtions > 2:
            # check overall significance
            if paired:
                if binary:
                    # cochran test
                    overall_pvalue = pg.cochran(
                        data=df, dv=meas_name, within="condition", subject="animal"
                    )["p-unc"].iloc[0]
                else:
                    # friedman test
                    overall_pvalue = pg.friedman(
                        data=df, dv=meas_name, within="condition", subject="animal"
                    )["p-unc"].iloc[0]
            else:
                # kruskal-wallis test
                overall_pvalue = pg.kruskal(data=df, dv=meas_name, between="condition")[
                    "p-unc"
                ].iloc[0]
        else:
            # we can do pairwise tests directly
            overall_pvalue = 0

        # do post-hoc tests
        if paired:
            # perform paired pairwise tests with pingouin (Wilcoxon)
            res = df.pairwise_tests(
                dv=meas_name,
                within="condition",
                parametric=False,
                subject="animal",
                nan_policy="pairwise",
            ).round(5)
        else:
            # perform unpaired test with pingouin (Mann-Whitney)
            res = df.pairwise_tests(
                dv=meas_name, between="condition", parametric=False
            ).round(5)

        # set p-values to 1 if the overall test was not significant
        if overall_pvalue > pthresh:
            res["p-unc"] = 1

        res["metric"] = meas_name  # keep track of the measurement name
        res["overall-p"] = overall_pvalue
        results[meas_name] = res

    return results


def select_consecutive_pvalues(
    df_pvalues: pd.DataFrame, conditions_list: list
) -> list[float]:
    """
    Get pvalues for pairs of consecutive conditions.

    Parameters
    ----------
    df_pvalues : pd.DataFrame
        DataFrame as returned by `pingouin.pairwise_tests()`.
    conditions_list : list
        List of conditions in the order they are plotted.

    Returns
    -------
    pvalues : list of float
        Consecutive condition pairwise pvalues.

    """
    pvalues = []

    for idx in range(len(conditions_list) - 1):
        cond1 = conditions_list[idx]
        cond2 = conditions_list[idx + 1]
        # get pair
        pval = df_pvalues.loc[
            (df_pvalues["A"] == cond1) & (df_pvalues["B"] == cond2), "p-unc"
        ]
        if len(pval) == 0:
            # try the reverse pair
            pval = df_pvalues.loc[
                (df_pvalues["B"] == cond1) & (df_pvalues["A"] == cond2), "p-unc"
            ]
        pvalues.append(pval.iloc[0])

    return pvalues


def get_quantif_metrics(
    df: pd.DataFrame, metrics_map: dict, range_map: dict, paired=False
):
    """
    Compute quantitative metrics during stimulation to compare groups (conditions).

    `df` must have in columns :
        - the features of interest,
        - a "condition" column that corresponds to the `conditions`.
        - a "time" column that is used to select stimulation epoch.
        - a "trialID" column that are unique to a time serie.
    Non-parametric pairwise tests are performed between conditions for each metrics.

    Parameters
    ----------
    df : pandas.DataFrame
    metrics_map : dict
        Mapping a feature to a function to apply to get the metric.
    range_map : dict
        Mapping a feature to time range in which the metric is computed.
    paired : bool, optional
        Whether to perform paired tests.

    Returns
    -------
    df_metrics : pandas.DataFrame
        DataFrame with metrics, one column for each feature, along with conditions and
        trial ID.
    pvalues : dict
        Map a metric name to a DataFrame with test result summary as returned by
        `pingouin.pairwise_tests()`.

    """
    metrics_df_list = []  # prepare DataFrame with metrics
    metric_names_list = []  # prepare list of metrics names
    pbar = tqdm(metrics_map.items())
    for feature, operations in pbar:
        pbar.set_description(f"Computing {feature} metrics")

        for metric, operation in operations.items():
            # select stimulation epoch
            df_stim = df[
                (df["time"] >= range_map[feature][metric][0])
                & (df["time"] < range_map[feature][metric][1])
            ]

            # get time vector
            time = df_stim["time"].unique()

            # group by conditions and trial ID
            df_gp = df_stim.groupby(["condition", "trialID"])

            # get grouped animal list
            s_animal = df_gp["animal"].unique().str.join("")

            # compute metrics for each time series
            s_metric = df_gp[feature].apply(operation, time)

            # make the metric full name
            metric_name = f"{feature}_{metric}"
            s_metric.name = metric_name  # add metric name
            metric_names_list.append(metric_name)  # store the metric name

            # merge
            df_metric = pd.concat([s_metric, s_animal], axis=1)

            # append to the rest
            metrics_df_list.append(
                df_metric.reset_index().set_index(["trialID", "condition"])
            )

    # concatenate DataFrames along rows (merge)
    df_metrics = pd.concat(metrics_df_list, axis=1).reset_index()
    # remove duplicated "animal" columns
    df_metrics = df_metrics.loc[:, ~df_metrics.columns.duplicated()].copy()

    # perform stat. tests
    if df_metrics["condition"].nunique() < 2:
        pvalues = {metric_name: pd.DataFrame() for metric_name in metric_names_list}
    else:
        pvalues = perform_stat_test(df_metrics, metric_names_list, paired=paired)

    return df_metrics, pvalues


def get_delays(
    df: pd.DataFrame,
    features: list,
    stim_time: list | tuple,
    nstd: float = 3,
    npoints: int = 3,
    maxdelay: float = 0.5,
    paired: bool = False,
):
    """
    Find delay of response with respect to stimulation onset. Also gets response.

    To determine motion onset :
    1. Find where the signal first deviates `nstd` times the pre-stim standard deviation
    from the pre-stim mean, after the stim onset.
    2. Take `npoints` from this point.
    3. Fit with a linear function.
    4. The delay is the intersection between the fit and the y = `nstd` * pre-stim std.

    Cases where the delay is discarded :
    - Condition never reached.
    - Fit with negative slope when the reference value is above
        pre-stim mean + `nstd` * pre-stim std.
    - Fit with positive slope when the reference value is below
        pre-stim mean - `nstd` * pre-stim std.

    Parameters
    ----------
    df : pandas.DataFrame
    features : list
        Keys in `df`, used to compute delays.
    stim_time : 2-elements list or tuple
        Stimulation onset and offset.
    nstd : float, optionnal
        Multiplier of standard deviation for threshold definition. Default is 3.
    npoints : int, optional
        Number of points to take for the fit. Default is 3.
    maxdelay : float, optional
        Maximum allowed delay, above which it is discarded, in the same units as time.
        Default is 0.5.
    paired : bool, optional
        Whether to perform paired tests.

    Returns
    -------
    df_response : pandas.DataFrame
        List of delays for each animals and each trials. NaN if no delay found.
    pvalues_delays, pvalues_response : dict
        Map a feature name to a DataFrame with test result summary as returned by
        `pingouin.pairwise_tests()`.

    """
    # group by trials and conditions
    df_group = df.groupby(["condition", "trialID"])

    df_response_feature = []
    pvalues_delays = {}
    pvalues_response = {}
    for feature in features:
        dfs = []  # initialize output
        for name, df_trial in df_group:
            # get pre-stim and post-time boolean masks
            pre_mask = df_trial["time"] < stim_time[0]
            post_mask = df_trial["time"] >= stim_time[0]

            premean = df_trial.loc[pre_mask, feature].mean()  # get pre-stim mean
            prestd = df_trial.loc[pre_mask, feature].std()  # get pre-stim std

            # thresholds to define reaction
            upper_threshold = premean + nstd * prestd
            lower_threshold = premean - nstd * prestd

            # find times it deviates nstd the pre-stim mean
            cond = (df_trial[feature] >= upper_threshold) | (
                df_trial[feature] <= lower_threshold
            )
            # select only post-stim values
            cond *= post_mask

            if not cond.any():
                # condition never reached
                onset = np.nan
            else:
                # check the value right before and after the stim onset
                last_value = df_trial.loc[pre_mask, feature].iloc[-1]
                first_value = df_trial.loc[post_mask, feature].iloc[0]

                # check if the signal was already above threshold before the stim onset
                if (
                    (last_value < lower_threshold) & (first_value < lower_threshold)
                ) | ((last_value > upper_threshold) & (first_value > upper_threshold)):
                    # re-center the thresholds
                    upper_threshold = last_value + nstd * prestd
                    lower_threshold = last_value - nstd * prestd

                    # update thresholding mask
                    cond = (df_trial[feature] > upper_threshold) | (
                        df_trial[feature] < lower_threshold
                    )
                    # select only post-stim values
                    cond *= post_mask

                # find first npoints consecutive values where condition is met
                consecutive_mask = cond.rolling(window=npoints).sum() == 3
                if not consecutive_mask.any():
                    # not enough values to proceed
                    onset = np.nan
                else:
                    first_index = consecutive_mask.idxmax() - 2
                    first_times_above = df_trial.loc[first_index:, "time"].to_numpy()[
                        :npoints
                    ]
                    first_values_above = df_trial.loc[first_index:, feature].to_numpy()[
                        :npoints
                    ]

                    # fit
                    p = np.polynomial.Polynomial.fit(
                        first_times_above, first_values_above, 1
                    )
                    # get slope
                    coef = p.convert().coef[1]
                    # determine if we consider lower or upper threshold
                    if first_values_above[0] >= upper_threshold:
                        # slope should be positive
                        if coef <= 0:
                            onset = np.nan
                        else:
                            # get intersection between fit and threshold
                            onset = (p - upper_threshold).roots()[0]
                    elif first_values_above[0] <= lower_threshold:
                        # slope should be negative
                        if coef >= 0:
                            onset = np.nan
                        else:
                            # get intersection between fit and threshold
                            onset = (p - lower_threshold).roots()[0]

            # check motion onset is after stim onset
            if onset <= stim_time[0]:
                # use 3 std directly instead
                onset = first_times_above[0]

            # get final delay
            delay = onset - stim_time[0]

            # check it does not exceed imposed value
            if delay > maxdelay:
                delay = np.nan

            # collect results
            dfs.append(
                {
                    "condition": name[0],
                    "trialID": name[1],
                    "delay": delay,
                    "filename": df_trial["filename"].iloc[0],
                    "animal": df_trial["animal"].iloc[0],
                }
            )

        df_resp = pd.DataFrame(dfs)  # convert to DataFrame
        df_resp["feature"] = feature  # add corresponding feature

        # compute response
        # 1 if a delay was computed, 0 otherwise
        df_resp["response"] = df_resp.loc[:, "delay"].notna() * 1
        # inverse of the delay, np.nan will stay np.nan
        df_resp["responsiveness"] = 1 / df_resp["delay"]
        # replace np.nan with 0 (no response)
        df_resp["responsiveness"] = df_resp["responsiveness"].fillna(0)

        # perform stat. tests
        if df_resp["condition"].nunique() < 2:
            pvalues_delays[feature] = pd.DataFrame()
            pvalues_response[feature] = pd.DataFrame()
        else:
            pval = perform_stat_test(df_resp, ["delay"], paired=paired)
            pvalues_delays[feature] = pval["delay"]
            pval = perform_stat_test(df_resp, ["response"], paired=paired)
            pvalues_response[feature] = pval["response"]

        df_response_feature.append(df_resp)

    df_response = pd.concat(df_response_feature).reset_index().drop(columns="index")

    return df_response, pvalues_delays, pvalues_response


def pvalue_to_stars(pvalue):
    """
    Convert p-values to stars.

    Parameters
    ----------
    pvalue : float

    Returns
    -------
    stars : str

    """
    if pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    elif pvalue > 0.05:
        return "ns"
    else:
        print("[Warning] pvalue computation errored.")
        return "error"


def add_stars_to_lines(pvalue, stim_time, ax, **kwargs):
    """
    Adds significance stars to specified ax based on given p-value.

    Parameters
    ----------
    pvalue : float
    stim_time : list or tuple
    ax : matplotlib axes
    **kwargs :
        Options passed to matplotlib.axes.Axes.text.

    Returns
    -------
    ax : matplotlib axes

    """
    if not pvalue:
        # pvalue was not computed
        return ax

    stars = pvalue_to_stars(pvalue)  # get stars

    # get text coordinates
    x = (stim_time[0] + stim_time[1]) / 2  # middle of stim. epoch
    y = ax.get_ylim()[1]  # top of the plot
    y = y - y * 0.025  # offset

    ax.text(
        x, y, stars, horizontalalignment="center", verticalalignment="top", **kwargs
    )

    return ax


def add_stars_to_bars(
    x: list | tuple,
    maxval: float,
    pvalue: float | str,
    ax: plt.Axes | None = None,
    line_kws: dict = {},
    text_kws: dict = {},
) -> plt.Axes:
    """
    Adds significance stars to bar plot based on given p-value.

    Parameters
    ----------
    x : list or tuple of 2 int
        Bars location, index-based.
    maxval : float
        Maximum value of what is plotted to adjust where the stars are added.
    pvalue : float or string
        If float, it is converted to a number of star. If str, it is written as is.
    tick_size : float
        Size of line tick, expressed as fraction of total height.
    ax : matplotlib axes
        Axes in which to add stars.
    line_kws : dict
        Options passed to matplotlib.axes.Axes.plot.
    text_kws : dict
        Options passed to matplotlib.axes.Axes.text.

    Returns
    -------
    ax : matplotlib Axes

    """
    if pvalue is None:
        return ax

    if not isinstance(pvalue, str):
        stars = pvalue_to_stars(pvalue)  # get stars

    axes_height = np.diff(ax.get_ylim())[0]  # axes height

    # get line coordinates
    xline1 = x[0]  # first bar
    xline2 = x[1]  # second bar
    yline_tick = maxval * 1.075  # where the ticks will be
    yline_top = yline_tick + 0.015 * axes_height  # where the line will be

    # add line
    ax.plot(
        [xline1, xline1, xline2, xline2],
        [yline_tick, yline_top, yline_top, yline_tick],
        **line_kws,
    )

    # get text coordinates
    xtxt = np.mean(x)  # should be between two bars
    ytxt = yline_top + 0.035 * axes_height  # add offset with respect to line

    # add text
    ax.text(
        xtxt,
        ytxt,
        stars,
        horizontalalignment="center",
        verticalalignment="top",
        **text_kws,
    )

    return ax


def add_patch(xmin: float, xmax: float, ax: plt.Axes, **kwargs) -> plt.Axes:
    """
    Plots a box from xmin to xmax, spanning all vertical space.

    Parameters
    ----------
    xmin, xmax : scalar
        Lower, Upper limits of the box to plot.
    ax : matplotlib Axes
        Axis in which to plot the box.
    **kwargs : passed to Axes.axvspan

    Returns
    -------
    ax : matplotlib Axes
        Handle to axes.

    """
    # plot patch
    ax.axvspan(xmin, xmax, zorder=-1, **kwargs)

    return ax


def add_arrows(
    ax: plt.Axes, right: bool = True, top: bool = True, size: float | None = None
) -> plt.Axes:
    """
    Add arrows to the axes' tips.

    Parameters
    ----------
    ax : matplotlib Axes
    right, top : bool, optional
        Add arrow on the x or y axis. Default is True.
    size : float, optional
        Size of the arrow marker. Default is None (matplotlib's default).

    Returns
    -------
    ax : matplotlib Axes

    """
    if right:
        ax.plot(1, 0, ">k", markersize=size, transform=ax.transAxes, clip_on=False)
    if top:
        ax.plot(0, 1, "^k", markersize=size, transform=ax.transAxes, clip_on=False)

    return ax


def set_nticks(ax: plt.Axes, nxticks: int | None, nyticks: int | None) -> plt.Axes:
    """
    Set number of ticks on axes.

    Parameters
    ----------
    ax : matplotlib Axes
    nxticks, nyticks : int or "auto" or None
        Number of ticks in x and y axes. Ignored if None.

    Returns
    -------
    ax : matplotlib Axes.

    """
    if nxticks:
        if nxticks == "auto":
            ax.locator_params(axis="x", nbins="auto")
        else:
            ax.locator_params(axis="x", nbins=nxticks + 1)

    if nyticks:
        if nyticks == "auto":
            ax.locator_params(axis="y", nbins="auto")
        else:
            ax.locator_params(axis="y", nbins=nyticks + 1)

    return ax


def on_pick(event, df: pd.DataFrame):
    """
    Callback function called when a line is clicked. Prints the corresponding animal
    and trial.

    Parameters
    ----------
    event : Event


    """
    # seaborn labels are "childXX" where XX is the index of the hue
    labelid = int(event.artist.get_label().split("_child")[1])
    label = df["filename"].unique()[labelid]

    print(label)


def nice_plot_serie(
    df: pd.DataFrame,
    x: str = "",
    y: str = "",
    xlabel: str = "",
    ylabel: str = "",
    conditions_order: list | None = None,
    pvalue: float | None = None,
    stim_time: list | tuple | None = None,
    ylim: list | None = None,
    ax: plt.Axes | None = None,
    plot_options: dict = {},
    kwargs_plot: dict = {},
) -> plt.Axes:
    """
    Nice plot of features, with mean and sem.

    Plot options are given in `kwargs_plot`. The latter must be returned by `set_plot_style`,
    and contain keys according to what was requested to be plotted.

    Parameters
    ----------
    df : pandas.DataFrame
    x, y : str
        Keys in `df`.
    xlabel, ylabel : str
        Labels of x and y axes.
    conditions_order : list or None
        If conditions are plotted, specifies in which order they are so.
    pvalue : float or None
        pvalue to display during stim.
    stim_time : 2-elements list or tuple or None
        Stimulation onset and offsets to draw the stimulation patch.
    ylim : list or None
        Limit of y axis.
    ax : matplotlib Axes
    plot_options : dict
        Plotting options.
    kwargs_plot : dict
        Dictionnary with plot style options.

    Returns
    -------
    ax : matplotlib Axes

    """
    if plot_options["plot_trials"]:
        # plot individual trials
        palette = df["trialID"].nunique() * [kwargs_plot["trial"]["color"]]
        ax = sns.lineplot(
            df,
            x=x,
            y=y,
            hue="trialID",
            estimator=None,
            errorbar=None,
            palette=palette,
            ax=ax,
            err_kws=kwargs_plot["sem"],
            **kwargs_plot["trial"],
        )

    if plot_options["plot_animal"]:
        # plot mean per animal
        palette = kwargs_plot["animal"]["color"]
        ax = sns.lineplot(
            df,
            x=x,
            y=y,
            hue="animal",
            estimator="mean",
            errorbar="se",
            palette=palette,
            ax=ax,
            err_kws=kwargs_plot["sem"],
            **kwargs_plot["animal"],
        )

    if plot_options["plot_condition"]:
        # plot mean per condition
        palette = kwargs_plot["condition"]["color"]
        ax = sns.lineplot(
            df,
            x=x,
            y=y,
            hue="condition",
            hue_order=conditions_order,
            estimator="mean",
            errorbar="se",
            palette=palette,
            ax=ax,
            err_kws=kwargs_plot["sem"],
            **kwargs_plot["condition"],
        )

    if plot_options["plot_pooled"]:
        # plot pooled mean
        ax = sns.lineplot(
            df[df["condition"].isin(plot_options["plot_pooled"])],
            x=x,
            y=y,
            estimator="mean",
            errorbar="se",
            ax=ax,
            err_kws=kwargs_plot["sem"],
            **kwargs_plot["pooled"],
        )

    # adjust limit
    if ylim:
        ax.set_ylim(ylim)

    # plot styling
    if kwargs_plot["arrow"]:
        # add arrows at axes tips
        ax = add_arrows(ax, size=kwargs_plot["arrow_size"])

    # determine if the legend should be shown
    if plot_options["plot_animal_monochrome"] or plot_options["plot_trials"]:
        # remove legend
        ax.get_legend().remove()

    # set number of axes ticks
    ax = set_nticks(ax, kwargs_plot["nxticks"], kwargs_plot["nyticks"])

    # set axes labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # add stimulation box
    if stim_time:
        ax = add_patch(stim_time[0], stim_time[1], ax, **kwargs_plot["stim"])

    ax = add_stars_to_lines(pvalue, stim_time, ax, fontsize="large")

    return ax


def nice_plot_metrics(
    df: pd.DataFrame,
    x: str = "condition",
    y: str = "",
    order: list = [],
    pvalues: list | float = 0,
    title="",
    ax: plt.Axes | None = None,
    kwargs_plot: dict = {},
) -> plt.Axes:
    """
    Nice plot of quantitative metrics to compare the stimulation effect on injected
    animals, compared to controls.

    Plot options are given in `kwargs_plot`. The latter must be returned by
    `set_plot_style`, and contain keys according to what was requested to be plotted.

    Parameters
    ----------
    df : pandas.DataFrame
    x, y : str
        Keys in `df`.
    order : list
        Order in which metrics will be plotted.
    pvalues : List
        List of p-values for consecutive conditions to plot stars. If 0, no stars will
        be plotted.
    title : str
        Axes title.
    ax : matplotlib Axes
    kwargs_plot : dict
        Dictionnary with plot style options.

    Returns
    -------
    ax : matplotlib Axes

    """
    ax = sns.barplot(
        df,
        x=x,
        y=y,
        hue=x,
        order=order,
        hue_order=order,
        estimator="mean",
        errorbar="se",
        ax=ax,
        **kwargs_plot["metrics"]["bars"],
    )

    # add data points
    # we need to pop one of its item and will be reused
    kws_pts = kwargs_plot["metrics"]["points"].copy()
    show_points = kws_pts.pop("show_points")
    if show_points:
        ax = sns.stripplot(
            df,
            x=x,
            y=y,
            hue=x,
            order=order,
            hue_order=order,
            legend=False,
            dodge=True,
            ax=ax,
            **kws_pts,
        )

    # add significance
    if pvalues:
        # get bar + errorbar value, sorting as sorted in the plot
        maxvals = (df.groupby(x)[y].mean() + df.groupby(x)[y].sem())[order].values

        for c, pvalue in enumerate(pvalues):
            ax = add_stars_to_bars(
                [c, c + 1],
                max(maxvals[c : c + 2]),
                pvalue,
                ax=ax,
                line_kws={"color": "k"},
                text_kws={"fontsize": "large"},
            )

    # remove axes labels
    ax.set_xlabel("")
    ax.set_ylabel("")

    # move x axis to 0
    ax.spines["bottom"].set_position("zero")

    # plot styling
    if kwargs_plot["arrow"]:
        # add arrows at axes tips
        ax = add_arrows(ax, right=False, size=kwargs_plot["arrow_size"])

    # set title
    ax.set_title(title)

    return ax


def nice_plot_bars(
    df: pd.DataFrame,
    x: str = "",
    y: str = "",
    hue: str = "",
    hue_order: list = [],
    pvalues: dict | None = None,
    xlabels: dict = {},
    ylabel: str = "",
    ax: plt.Axes | None = None,
    kwargs_plot: dict = {},
) -> plt.Axes:
    """
    Nice bar plot of `y`.

    Parameters
    ----------
    df : pandas.DataFrame
    x, y, hue : str
        Keys in `df`.
    hue_order : list
        Order in which to plot the hues.
    pvalues : dict
        Mapping a `x` to a list of pvalues for consecutive `hue`.
    xlabels : dict
        Mapping names found in 'x' in `df` to another values.
    ylabel : str
    ax : matplotlib Axes

    Returns
    -------
    ax : matplotlib Axes

    """
    # modify x values to change their labels in the plot
    df_plot = df.copy()
    df_plot["feature"] = df_plot["feature"].map(xlabels)

    ax = sns.barplot(
        df_plot,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        estimator="mean",
        errorbar="se",
        ax=ax,
        **kwargs_plot["delays"]["bars"],
    )

    # add data points
    kwplot = kwargs_plot["delays"]["points"].copy()
    show_points = kwplot.pop("show_points")
    if show_points:
        ax = sns.stripplot(
            df_plot,
            x=x,
            y=y,
            hue=hue,
            hue_order=hue_order,
            legend=False,
            dodge=True,
            ax=ax,
            **kwplot,
        )

    # add significance
    if pvalues:
        # get offset for each bar at same x (from seaborn)
        n_levels = df[hue].nunique()  # number of hue
        width = 0.8  # 0.8 is default width in sns.barplot()
        each_width = width / n_levels
        offsets = np.linspace(0, width - each_width, n_levels)
        offsets -= offsets.mean()

        for xline_center, xp in enumerate(df[x].unique()):
            # select data
            dfpval = df.loc[df[x] == xp, :]
            pvalue = pvalues[xp]

            if not pvalue:
                continue

            # get bar + errorbar value, sorting as sorted in the plot
            maxvals = (dfpval.groupby(hue)[y].mean() + dfpval.groupby(hue)[y].sem())[
                hue_order
            ].values

            for c, pval in enumerate(pvalue):
                xline_0 = xline_center + offsets[c]
                xline_1 = xline_center + offsets[c + 1]
                ax = add_stars_to_bars(
                    [xline_0, xline_1],
                    max(maxvals[c : c + 2]),
                    pval,
                    ax=ax,
                    line_kws={"color": "k"},
                    text_kws={"fontsize": "large"},
                )

    # add/remove axes labels
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)

    # plot styling
    if kwargs_plot["arrow"]:
        # add arrows at axes tips
        ax = add_arrows(ax, right=False, size=kwargs_plot["arrow_size"])

    return ax


def nice_plot_raster(
    df: pd.DataFrame,
    x: str = "",
    y: str = "",
    features: list | tuple = [""],
    conditions: list | tuple = [""],
    xlabel: str = "",
    clabels: dict = {},
    kwargs_plot: dict = {},
    quantile: float = 0.99,
    **kwargs,
) -> plt.Figure:
    """
    Display time series as a temporal raster plot.

    The input DataFrame should be generated with `process_animal`.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    x : str
        Key in `df`. Should allow to extract time series (typically, "time").
    y : str
        Key in `df`. Should allow to extract trials (typically "trialID").
    features : list | tuple
        Keys in `df` that will be plotted in different subplots, sharing their x-axis.
    conditions : list | tuple
        Values in `df["condition"]` that will be plotted, sharing their colorbar.
    xlabel : str
        Label of x axis.
    clabels : dict
        Mapping feature to colorbar label.
    kwargs_plot : dict
        Dictionnary with plot style options.
    quantile : float, optional
        Quantile to use to get colormap range. Default is 0.99.
    **kwargs : passed to pyplot.pcolormesh.

    Returns
    -------
    fig : matplotlib Figure

    """
    # select data
    df_in = df.copy()
    df_in = df_in[df_in["condition"].isin(conditions)]

    # get shapes
    nconditions = len(conditions)
    nfeatures = len(features)
    time = df_in[x].unique()
    ntimes = len(time)

    # get values range
    cranges = [
        [df[feature].quantile(1 - quantile), df[feature].quantile(quantile)]
        for feature in features
    ]

    # prepare figure
    fig, axs = plt.subplots(
        nfeatures, nconditions, sharex=True, squeeze=False, figsize=(15, 8)
    )
    fig.subplots_adjust(right=0.9)

    for idx_condition in range(nconditions):
        condition = conditions[idx_condition]
        df_cond = df_in[df_in["condition"] == condition]
        trials = df_cond[y].unique()
        ntrials = len(trials)

        # get indices of change of animals
        ser = df_cond.groupby([y])["animal"].unique()
        ind_animals = np.where(ser.ne(ser.shift().bfill()))[0]

        for idx_feature in range(nfeatures):
            feature = features[idx_feature]

            # prepare raster data
            data = np.reshape(df_cond[feature].values, (ntrials, ntimes))

            # raster plot
            p = axs[idx_feature, idx_condition].pcolormesh(
                time,
                trials,
                data,
                vmin=cranges[idx_feature][0],
                vmax=cranges[idx_feature][1],
                **kwargs,
            )

            # add animal delimitation
            _ = [
                axs[idx_feature, idx_condition].axhline(
                    loc - 0.5, linewidth=1.5, color="#ec1b8a"
                )
                for loc in ind_animals
            ]

            # add colorbar if it's the last panel on the right
            if idx_condition == nconditions - 1:
                axpos = (
                    axs[idx_feature, idx_condition]
                    .get_position()
                    .get_points()
                    .flatten()
                )
                axheight = axpos[3] - axpos[1]
                cax = fig.add_axes((axpos[2] + 0.01, axpos[1], 1 / 75, axheight))
                plt.colorbar(p, label=clabels[feature], cax=cax)

            # add xlabel if it's the last panel on the bottom
            if idx_feature == nfeatures - 1:
                axs[idx_feature, idx_condition].set_xlabel(xlabel)

            # add title if it's the first panel on the top
            if idx_feature == 0:
                axs[idx_feature, idx_condition].set_title(condition)

            # adjust style
            axs[idx_feature, idx_condition] = set_nticks(
                axs[idx_feature, idx_condition],
                kwargs_plot["nxticks"],
                None,
            )
            axs[idx_feature, idx_condition].set(yticklabels=[])
            axs[idx_feature, idx_condition].tick_params(left=False)
            axs[idx_feature, idx_condition].spines.right.set_visible(True)
            axs[idx_feature, idx_condition].spines.top.set_visible(True)

    return fig


def process_features(df_features: pd.DataFrame, cfg, paired: bool = False):
    """
    Get data for `plot_all_figures()` from the features DataFrame.

    Computes pvalue during stimulation, in-stim quantifying metrics and corresponding
    pvalues and responses with delays.

    Parameters
    ----------
    df_features : pd.DataFrame
    cfg : Config
    paired : bool

    Returns
    -------
    pvalues_stim : dict
    df_metrics : pd.DataFrame
    pvalues_metrics : dict
    df_response : pd.DataFrame
    pvalues_delays : dict
    pvalues_response : dict

    """

    # get pre/during stim p-values
    pvalues_stim = {
        feature: get_pvalue_timeserie(df_features, feature, cfg.stim_time)
        for feature in cfg.features.keys()
    }

    # get in-stim quantitative metric
    df_metrics, pvalues_metrics = get_quantif_metrics(
        df_features, cfg.features_metrics, cfg.features_metrics_range, paired=paired
    )

    # delays before motion onset for each feature
    df_response, pvalues_delays, pvalues_response = get_delays(
        df_features,
        cfg.features,
        cfg.stim_time,
        nstd=cfg.nstd,
        npoints=cfg.npoints,
        maxdelay=cfg.maxdelay,
        paired=paired,
    )
    df_response["delay"] = df_response["delay"] * 1000  # convert to ms

    return (
        pvalues_stim,
        df_metrics,
        pvalues_metrics,
        df_response,
        pvalues_delays,
        pvalues_response,
    )


def plot_all_figures(
    df_features: pd.DataFrame,
    pvalues_stim: dict | None,
    df_metrics: pd.DataFrame,
    pvalues_metrics: dict,
    df_response: pd.DataFrame,
    pvalues_delays: dict,
    pvalues_response: dict,
    plot_options: dict,
    conditions_list: list,
    cfg,
) -> list[plt.Figure]:
    """
    Wraps plotting functions.

    Parameters
    ----------
    df_features : pd.DataFrame
        DataFrame with time series as returned by `process_animal()` or
        `process_directory()`.
    pvalues_stim : dict
        Maps a feature to a pvalue for time series mean during stim.
    df_metrics : pd.DataFrame
        DataFrame with metrics quantifying in-stim change, as returned by
        `process_features()`.
    pvalues_metrics : dict
        Maps a metric name to stat. tests performed with `pingouin`, as returned by
        `process_features()`.
    df_response : pd.DataFrame
        DataFrame with delays, response and responsiveness as returned by
        `process_features()`.
    pvalues_delays : dict
        Maps a feature to a stat. tests performed with `pingouin`, as returned by
        `process_features()`.
    pvalues_response: dict
        Maps a feature to a stat. tests performed with `pingouin`, as returned by
        `process_features()`.
    plot_options : dict
        Plotting options (see example script).
    conditions_list : list
        List of conditions to plot in which order.
    cfg : Config
        Configuration object.

    Returns
    -------
    list[plt.Figure]
        List of all generated figures.

    """

    # select data
    if plot_options["plot_condition_off"]:
        # remove conditions that will not be plotted
        df_features_plt = df_features[
            ~df_features["condition"].isin(plot_options["plot_condition_off"])
        ]
    else:
        df_features_plt = df_features  # take all conditions

    # prepare style
    kwargs_plot = setup_plot_style(
        style_file=plot_options["style_file"],
        plot_animal_monochrome=plot_options["plot_animal_monochrome"],
        nanimals=len(df_features["animal"].unique()),
    )

    # - Features
    figs_features = []
    pbar = tqdm(cfg.features.keys())
    for feature in pbar:
        if feature in cfg.features_off:
            continue

        pbar.set_description(f"Plotting {feature}")

        # prepare figure
        nmetrics = len(cfg.features_metrics[feature])
        nrows = 1
        ncols = 3 + nmetrics  # 2 for time series, then 1 for each metric
        fig = plt.figure(figsize=kwargs_plot["figsize"])
        axs = []
        # create axis for time serie
        ax = fig.add_subplot(nrows, ncols, (1, 3))
        axs.append(ax)
        # create axes for in-stim metrics
        for idx, sharey in zip(
            range(4, ncols + 1), cfg.features_metrics_share[feature].values()
        ):
            ax = fig.add_subplot(nrows, ncols, idx, sharey=axs[0] if sharey else None)
            axs.append(ax)

        # time series
        if feature in cfg.features_ylim:
            ylim = cfg.features_ylim[feature]
        else:
            ylim = None
        nice_plot_serie(
            df_features_plt,
            x="time",
            y=feature,
            xlabel=cfg.xlabel_line,
            ylabel=cfg.features_labels[feature],
            conditions_order=conditions_list,
            pvalue=pvalues_stim[feature],
            stim_time=cfg.stim_time,
            ylim=ylim,
            ax=axs[0],
            plot_options=plot_options,
            kwargs_plot=kwargs_plot,
        )
        if cfg.xlim:
            axs[0].set_xlim(cfg.xlim)

        # add callback on lines
        fig.canvas.mpl_connect("pick_event", partial(on_pick, df=df_features_plt))

        # metrics bars
        for metric, ax in zip(cfg.features_metrics[feature], axs[1::]):
            # build metric name
            metric_name = f"{feature}_{metric}"
            # get consecutive pvalues between conditions
            pvals = select_consecutive_pvalues(
                pvalues_metrics[metric_name], conditions_list
            )
            # plot bars
            nice_plot_metrics(
                df_metrics,
                x="condition",
                y=metric_name,
                order=conditions_list,
                pvalues=pvals,
                title=metric,
                ax=ax,
                kwargs_plot=kwargs_plot,
            )

        plt.show()
        figs_features.append(fig)

    # - Raster plot (here all conditions are plotted)
    print("Plotting raster plot...", end="", flush=True)
    fig_raster = nice_plot_raster(
        df_features,  # plot all conditions
        x="time",
        y="trialID",
        features=list(cfg.features.keys()),
        conditions=conditions_list,
        xlabel=cfg.xlabel_line,
        clabels=cfg.features_labels,
        kwargs_plot=kwargs_plot,
        quantile=0.99,
    )
    plt.show()
    print("\t Done.")

    # - Delays
    print("Plotting delays...", end="", flush=True)
    fig_delay, axd = plt.subplots(figsize=kwargs_plot["figsize"])
    df_response_plt = df_response[~df_response["feature"].isin(cfg.features_off)]
    pval_delays_plt = {
        feature: select_consecutive_pvalues(pvalues_delays[feature], conditions_list)
        for feature in pvalues_delays.keys()
    }
    nice_plot_bars(
        df_response_plt,
        x="feature",
        y="delay",
        hue="condition",
        hue_order=conditions_list,
        pvalues=pval_delays_plt,
        xlabels=cfg.features_labels,
        ylabel="delay (ms)",
        ax=axd,
        kwargs_plot=kwargs_plot,
    )
    print("\t Done.")

    # - Response
    print("Plotting response...", end="", flush=True)
    fig_response, axr = plt.subplots(figsize=kwargs_plot["figsize"])
    pval_response_plt = {
        feature: select_consecutive_pvalues(pvalues_response[feature], conditions_list)
        for feature in pvalues_response.keys()
    }

    nice_plot_bars(
        df_response_plt,
        x="feature",
        y="response",
        hue="condition",
        hue_order=conditions_list,
        pvalues=pval_response_plt,
        xlabels=cfg.features_labels,
        ylabel="response rate",
        ax=axr,
        kwargs_plot=kwargs_plot,
    )
    print("\t Done.")

    # - Responsiveness
    print("Plotting responsiveness...", end="", flush=True)
    fig_rspness, axrs = plt.subplots(figsize=kwargs_plot["figsize"])
    nice_plot_bars(
        df_response_plt,
        x="feature",
        y="responsiveness",
        hue="condition",
        hue_order=conditions_list,
        xlabels=cfg.features_labels,
        ylabel="responsiveness (ms$^{-1}$)",
        ax=axrs,
        kwargs_plot=kwargs_plot,
    )
    print("\t Done.")

    return figs_features, fig_raster, fig_delay, fig_response, fig_rspness


def process_directory(
    modality: str,
    path_to_configs: str,
    directory: str,
    animals: tuple | list,
    conditions: dict,
    plot_options: dict,
    outdir: str | None = None,
    paired: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process the directory.

    Looks for CSV file, link them to animals, get quantitative features and plot
    trial-averaged sequences.

    Parameters
    ----------
    modality : str
        Name of the configuration module.
    path_to_configs : str
        Full path to the configuration modules.
    directory : str
        Input directory.
    animals : list, tuple
        List of animals to perform the analysis on.
    conditions : dict
        Conditions and how to get them from the file name.
    plot_options : dict
        Plot options.
    outdir : str, optional
        If not None, saves figures there. Default is None.
    paired : bool, optional
        Whether to perform paired tests, default is False.

    Returns
    -------
    df_features, df_metrics, df_response : pd.DataFrame
        Resp. contain the features time series, the in-stim quantifying metrics and the
        delays and responsiveness.

    """
    # --- Prepare data
    # look for settings.toml file
    settings_file = os.path.join(directory, "settings.toml")

    cfg = get_config(modality, path_to_configs, settings_file)

    # try with .h5 files
    files_list = [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.endswith(".h5")
    ]

    if len(files_list) == 0:
        # if no h5 files found, look for .csv files instead.
        files_list = [
            os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if filename.endswith(".csv")
        ]

    # create files that logs used and dropped trials
    if outdir:
        # create output directory if needed
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        dropfile = os.path.join(outdir, "dropped.txt")
        open(dropfile, "w").close()
        usefile = os.path.join(outdir, "used.txt")
        open(usefile, "w").close()
    else:
        dropfile = None
        usefile = None

    # --- Process data
    df_align_list = []  # prepare list of DataFrames
    pbar = tqdm(animals)
    for animal in pbar:
        pbar.set_description(f"Processing animal {animal}")
        df_align_list.append(
            process_animal(
                files_list, animal, conditions, cfg, dropfile=dropfile, usefile=usefile
            )
        )
    # concatenate in a single DataFrame
    df_features = pd.concat(df_align_list).reset_index(drop=True)

    # derive metrics, pvalues, delays and response
    (
        pvalues_stim,
        df_metrics,
        pvalues_metrics,
        df_response,
        pvalues_delays,
        pvalues_response,
    ) = process_features(df_features, cfg, paired=paired)

    # --- Plot everything
    figs_features, fig_raster, fig_delay, fig_response, fig_rspness = plot_all_figures(
        df_features,
        pvalues_stim,
        df_metrics,
        pvalues_metrics,
        df_response,
        pvalues_delays,
        pvalues_response,
        plot_options,
        list(conditions.keys()),
        cfg,
    )

    # --- Save results
    if outdir:
        # save figures
        for fig, feature in zip(figs_features, cfg.features):
            pbar.set_description(f"Saving figure {feature}")
            fig.savefig(os.path.join(outdir, f"fig_{feature.replace('_', '')}.svg"))
        fig_delay.savefig(os.path.join(outdir, "fig_delays.svg"))
        fig_rspness.savefig(os.path.join(outdir, "fig_responsiveness.svg"))
        fig_response.savefig(os.path.join(outdir, "fig_response.svg"))
        fig_raster.savefig(os.path.join(outdir, "fig_raster.svg"))

        # save tables
        df_features.to_csv(os.path.join(outdir, "features.csv"), index=False)
        df_metrics.to_csv(os.path.join(outdir, "metrics.csv"), index=False)
        df_response.to_csv(os.path.join(outdir, "response.csv"), index=False)
        for metric, pval in pvalues_metrics.items():
            pval.to_csv(os.path.join(outdir, f"stats_{metric}.csv"), index=False)
        for feature in pvalues_delays.keys():
            pvalues_delays[feature].to_csv(
                os.path.join(outdir, f"stats_delays_{feature}.csv"), index=False
            )
            pvalues_response[feature].to_csv(
                os.path.join(outdir, f"stats_response_{feature}.csv"), index=False
            )

        # save parameters (only the last pixel size used will be written)
        cfg.write_parameters_file(outdir, name="analysis_parameters.toml")

    else:
        print(
            (
                "[Warning] No output directory specified : "
                "parameters and dropped trials were not written!"
            )
        )

    return df_features, df_metrics, df_response
