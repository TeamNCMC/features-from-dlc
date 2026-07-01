"""
Script to quantify behavior from DeepLabCut tracking using the features_from_dlc package.

Specify each entry, reading carefully what they do, then run the script with the 'ffd'
conda environment activated.

Works with features_from_dlc v2026.07.01

"""

import os
import matplotlib.pyplot as plt
import pandas as pd

import features_from_dlc as ffd

# --- Parameters ---
# - Inputs
directory = "/path/to/dlc/files"  # full path to the folder with files to analyze
configs_path = "../configs/"  # full path to where the configs files are
modality = "openfield"  # name of the configuration file (without .py)

# - Animals
# Only files beginning by those will be processed. If only one, write as ("xxx",)
animals = ("animal0", "animal1", "animal2")

# - Groups
# This must be a dicp:\Nella\scripts\ffd_vAlexis\nose_50ms.pytionnary {key: values}.
# "key" is the name of the condition that will appear in the graphs -- they must be a
# unique string ("a"). "values" must be a list (["a", "b"] or ["a"]). It is used to
# filter file names, eg. files with these elements in their name will be associated to
# this condition. First, we check if the filename begins with those filters, if it does,
# it's assigned to the corresponding condition, whether there's something else eleswhere
# in the file name. See get_condition() function for examples.
conditions = {
    "condition1": ["animal0"],
    "condition2": ["something"],
    "condition3": ["something_else"],
}

# Choose whether the conditions are paired, eg. if the same subject appears in several
# conditions, in which case, paired significance tests will be performed. In that case,
# all pairs of conditions should involve at least one time the same animal. The other
# case is unpaired tests (False).
paired_tests = False
 
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
    plot_animal=True,  # whether to plot mean and sem per animal
    plot_animal_monochrome=True,  # whether to plot mean per animal in the same color
    plot_condition_off=None,  # conditions NOT to be plotted, list or None
    style_file=style_file,  # full path to the config_plot.toml file
    plot_sem_per_animal =True, # whether to plotsem per animal
    per_animal=True,
)

# Delay onset computation: 
# - if th_with_3_std is True, refine the onset estimate using a linear fit
# - if th_with_3_std is False, use the first time point above the threshold directly as onset
th_with_linear_fit = False

# Call the processing function
df, metrics, response = ffd.process_directory(
    modality,
    configs_path,
    directory,
    animals,
    conditions,
    plot_options,
    th_with_linear_fit,
    outdir=outdir,
    paired=paired_tests,
)

# # Alternatively, use already generated features.csv file
#cfg = ffd.get_config(modality, configs_path, None)  # get config
#features = pd.read_csv()  # load features

# # compute values
#pvalues_stim, df_metrics, pvalues_metrics, pvalues_metrics_mean, df_response, pvalues_delays, pvalues_response, df_metrics_mean = ffd.process_features(features, computation_option, cfg)

#conditions_list = ["MdV", "Gi", "PnC", "WT"]

# # plot (figures won't be saved automatically)
#figs_features, fig_raster, fig_delay, fig_response, fig_rspness = ffd.plot_all_figures(
#    features,
#    pvalues_stim,
#    df_metrics,
#    df_metrics_mean,
#    pvalues_metrics,
#    pvalues_metrics_mean,
#    df_response,
#    pvalues_delays,
#    pvalues_response,
#    plot_options,
#    conditions_list,
#    cfg,
#)
#for fig in figs_features:
#    fig.show()

#for fig in [fig_raster, fig_delay, fig_response, fig_rspness]:
#    fig.show()

#plt.show(block=True)
