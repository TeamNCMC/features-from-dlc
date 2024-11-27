"""
Script to quantify behavior from DeepLabCut tracking using the features_from_dlc package.

Specify each entry, reading carefully what they do, then run the script with the 'ffd'
conda environment activated.

Works with features_from_dlc v2024.11.27

"""

import os

import pandas as pd

import features_from_dlc as ffd

# --- Parameters ---
# - Inputs
directory = "/path/to/dlc/files"  # full path to the folder with files to analyze
configs_path = "../configs/"  # full path to where the configs files are
modality = "openfield"  # name of the configuration file (without .py)

# - Animals
# Only files beginning by those will be processed. If only one, write as ("xxx",)
animals = ("animal0", "animal1")

# - Groups
# This must be a dictionnary {key: values}.
# "key" is the name of the condition that will appear in the graphs -- they must be a
# unique string ("a"). "values" must be a list (["a", "b"] or ["a"]). It is used to
# filter file names, eg. files with these elements in their name will be associated to
# this condition. First, we check if the filename begins with those filters, if it does,
# it's assigned to the corresponding condition, whether there's something else eleswhere
# in the file name. See get_condition() function for examples.
conditions = {
    "condition1": ["mouse0"],
    "condition2": ["identifier"],
    "condition3": ["something_else"],
}

# - Outputs
# Directory where figures and mean time series will be saved
# do not save anything :
# outdir = None
# create a "results" subdirectory :
outdir = os.path.join(directory, f"results_{"-".join([animal for animal in animals])}")
# choose directly where to save results :
# outdir = /path/to/custom/directory

# - Options
# Full path to the config_plot.toml file
style_file = os.path.join(configs_path, "config_plot.toml")
# Here, we choose what to plot. When "condition" is mentionned, it means the string must
# be exactly one of the conditions defined in CONDITIONS above.
plot_options = dict(
    plot_pooled=False,  # conditions whose trials are pooled to plot mean
    plot_trials=False,  # whether to plot individual trials
    plot_condition=True,  # whether to plot mean and sem per condition
    plot_animal=False,  # whether to plot mean and sem per animal
    plot_animal_monochrome=True,  # whether to plot mean per animal in the same color
    plot_condition_off=None,  # conditions NOT to be plotted, list or None
    plot_delay_list=[
        "condition2",
        "condition3",
    ],  # delays will be plotted only for those
    style_file=style_file,  # full path to the config_plot.toml file
)

# Call the processing function
df, metrics, response = ffd.process_directory(
    modality, configs_path, directory, animals, conditions, plot_options, outdir=outdir
)

# # Alternatively, use already generated features.csv file
# cfg = ffd.get_config(modality, configs_path, None)  # get config
# features = pd.read_csv(os.path.join(outdir, "features.csv"))  # load features

# # compute values
# pvalues_stim, df_metrics, pvalues_metrics, df_response = ffd.process_features(
#     features, conditions, cfg
# )

# # plot (figures won't be saved automatically)
# ffd.plot_all_figures(
#     features,
#     pvalues_stim,
#     df_metrics,
#     pvalues_metrics,
#     df_response,
#     plot_options,
#     conditions,
#     cfg,
# )
