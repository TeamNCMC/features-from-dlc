# Analyse DeepLabCut outputs
This repository contains a main script called `features_from_dlc` that is used to compute and plot behavioral metrics from DeepLabCut tracking files.

You'll also find some utility scripts in the scripts folder, as well as separate notebooks (.ipynb files) in the notebooks directory.

Jump to :
- [Install requirements](#quick-start)
- [The `video_cutter` script](#using-the-video_cutter-script)
- [The `features_from_dlc` script](#the-features_from_dlc-script)

##  Quick start
To use the scripts and notebooks, you first need to install some things. If conda is already installed, ignore steps 1-2. Those steps need to be done only once.

1. Install [miniconda3](https://docs.anaconda.com/miniconda/#id2) as user, add conda to PATH and make it the default interpreter.
2. Open a terminal (PowerShell in Windows) and run `conda init`. Restart the terminal.
3. Download, unzip and put this repository in a smart location (eg. `~/programs` or whatever).
4. Browse to this location from the terminal :
  ```bash
  cd /path/to/the/smart/location/features-from-dlc
  ```
5. Create a virtual environment named "ffd" with the required packages, using the provided `environment.yml file` : 
  ```bash
  conda env create -f environment.yml
  ```

You should be ready to use the scripts and notebooks !

To use the script, have a look at the instructions :
- [`video_cutter.py`](#using-the-video_cutter-script)
- [`features_from_dlc.py`](#the-features_from_dlc-script)

To use the notebooks, two options :
- Use an IDE with Jupyter notebooks support such as [Visual Studio Code](https://code.visualstudio.com/download). Install the Python and Jupyter extensions (the squared pieces on the left panel). Open the notebook with vscode, on the top right you should be able to select a kernel : choose "ffd".
- Use Jupyter directly in its web browser interface : from the terminal, activate the conda environment : `conda activate ffd`, then launch Jupyter : `jupyter lab /path/to/the/notebooks/notebook.ipynb`

##  Using the video_cutter script
This script (`/scripts/video_cutter.py`) is used to cut long recording containing multiple stimulations into single-stimulation clips.

It uses the video file and a corresponding stimulation trace stored as a text or binary file.

To use it, first install required dependencies following steps 1-5 in the [Quick start](#quick-start) section.  
Then, you need to install ffmpeg on your system. If it is not the case already you can follow this for Windows :  
1. Download the latest "essential" build from [here](https://www.gyan.dev/ffmpeg/builds/). Unzip it somewhere relevant (in a "Programs" directory or something), and copy the path to the `bin` folder (making sure it contains both 'ffmpeg.exe' and 'ffprobe.exe').
2. Paste this path in the `FFMPEG_BIN_DIR` variable. In the event ffmpeg is added to your PATH (eg. `ffmpeg` is recognised in a terminal), you can leave `FFMPEG_BIN_DIR` set to an empty string (`""`).
3. Fill in the parameters according to your need.
3. From the terminal with the ffd environment activated, browse to the script location :  
```bash
conda activate ffd
cd path/to/features-from-dlc/scripts/
```

And run the script :  
```bash
python video_cutter.py path/to/your/videos
```

The script is built as a command line interface (cli), for detailed usage, use :  
```bash
python video_cutter --help
```

The script processes all videos found in the input directory that have a txt/csv/bin file with the same name. The latter is read to determine the stimulation onsets. Extracted clips are created in {video-name}-cropped folder and are numbered from 0 to the number of stimulations found in the stimulation file.

### Notes
- The format of txt files exported from Labscribe depends on its version... Sometimes the values are separated by commas (`,`), sometimes tabulations. To be sure, open the file with a text editor and see if there are "," or big spaces between values on a row. Edit the `SEP` parameter accordingly in the script.
- Again with Labscribe, it's unclear when or which versions creates a header to this text file (names for each columns). The two should work : if there are non-numeric values on the first line of the file, it will be considered as a header and discarded.
- Parameters (time before and after stimulation onset, ...) are set as variables directly in the script, but you can override them by passing options to the script from the command line, like so :  
`python video-cutter.py path/to/videos --sep \t`

## The `features_from_dlc` script

Current version : 2024.09.26

### Introduction
This script is meant to be used to compute and display features from DeepLabCut (DLC) tracking files.
It will process a bunch of h5 or csv files created by DLC, computing so called "features" that are arbitrarily defined by the user and displaying them in nice graphs : time series with averages and errors, corresponding bar plots that quantifies a change during a specified epoch (typically, an optogenetic stimulation).  
It is intended to be modular : the main script (`features_from_dlc.py`) merely loads DLC files, computes features, averages them per condition and plots them. So-called configuration files are plugged into it and specifies _how_ the features are computed from the bodyparts tracked in DLC.  
Anyone can write its own configuration file to compute any required features (such as speed, body angle, jaw opening, you name it), as long as the original syntax is respected.

#### Tip
You don't have to read this document entirely to get started, you can jump to the [Getting started](#getting-started), [Requirements](#requirements) or [Usage](#usage) sections.

In some extent, the script itself is self-documented, which means _a lot_ of information on how things work are written as comment (`#`) or docstrings (`"""`) within the script.

#### Use case
The use case is :
- I have DLC files that tracks bodyparts "a", "b" and "c" (eg. "nose", "body", "tail").
- I want to compute and display features "x" and "y" (eg. "speed", "body angle"), that are derived from "a", "b" and "c".
- Those features should be averaged and compared across conditions and/or animals.
- There is a stimulation in my experiment and I want to quantify the change of behavior during the stimulation.

#### Modularity
The attempt to make this modular is the idea that the principle is always the same : load files, compute features, average across animals and conditions and plot the result. What actually differs from experiment to experiment and people to people is _how_ to compute the required features and the experimental conditions, such as the pixel size in the videos, the framerate and the timings of the stimulation. So all the later are defined in a separate configuration file that is experiment- and people-specific and plugged into the main script.

### How-to  
#### Getting started
Follow steps 1-5 in the [Quick start](#quick-start) section. Then, the idea is to edit the script and configuration files before running it your data. In principle you can do that with any text editor, but it is recommended to use an IDE for ease of use. You can use any of your liking, below is explained how to use Visual Studio Code.

##### Visual Studio Code
It's easier to use as conda is nicely integrated and it is made easy to switch between environments.
1. Install [vscode](https://code.visualstudio.com/download) (it does not require admin rights).
2. Install Python extension (squared pieces in the left panel).
3. Open the features_from_dlc.py script. In the bottom right corner, you should see a "conda" entry : click on it and select the ffd conda environment. To run the script, click on the Play item on the top right.

#### Requirements
You need to have tracked your video clips with DeepLabCut and saved the output files (either .h5 or .csv files). One file corresponds to one and only one trial, so you might need to split your original videos into several short clips around the stimulation onsets and offsets beforehand. This can be done with the script provided in the scripts folder. All files analyzed together must :
- be individual trials with the same duration, the same pixel size, and the stimulation must occur at the same timings,
- all be in the same directory,
- have consistent file names. In particular, those file names must :
  - start with the animal ID,
  - contains _something_ that allows to group those files into different categories. Those are called conditions. For instance, if you want to compare several animals at different laser power, their file names must contain something that identify that power, for instance all trials corresponding to 10mW will have "10mW" somewhere in their file names, while all trials corresponding to "20mW" will have "20mW" somewhere in their file names. Therefore, in that case, a file can not have both "10mW" and "20mW" in its filename. Of course, this applies for any conditions. Note that it means it can be limiting as you can't analyze 5mW and 25mW as different conditions (because "5mW" will appear in both groups).

You also need a configuration file. It defines the features one wants to extract, the metrics that quantify the change during stimulation, the thresholds used to filter incorrect tracks and some display parameters (axes names). See [The configuration file](#the-configuration-file).

Optionnaly, you can have a settings.toml file next to DLC files to analyze. It specifies the experimental settings (timings and pixel size). If the file does not exist, default values from the configuration file will be used instead. See [The settings.toml file](#the-settingstoml-file).

#### Usage
1. Read the docstring at the top of the script.
2. Fill the `--- Parameters ---` section. This includes :
   - `MODALITY` : a string (`"abc"`). It corresponds to a configuration file. It will look for a python file named `abc.py` in the configs directory.
   - `ANIMALS` : a tuple (`(a, b, c)`, or `(a, )` for single element). Only files beginning by those will be processed.
   - `CONDITIONS` : a dictionary (`{key1: values1, key2: values2}`). This maps a condition name ("key1", "key2" will appear on the graphs) to a _filter_. This filter is used to assign a file to a condition based on its file name. The beginning of the file name has priority, if it begins by something in the values, it is assigned to the corresponding condition, whether there is another match somewhere else in the file name.  
   Examples :  
       `conditions = {"control": ["animal70"], "low": ["10mW"], "high": ["20mW"]}`  
       filename -> condition :  
       animal70_10mW_blabla.h5 -> "control"  
       animal70_20mW_blabla.h5 -> "control"  
       animal71_10mW_blabla.h5 -> "low"  
       animal81_20mW_blabla.h5 -> "high"  
3. Fill the `- Options` section. Those are the display options, eg. what to plot and how :
   - `PLOT_POOLED`: `None` or a list of conditions (`["a"]` or `["a", "b"]`). Conditions whose trials are pooled to plot a pooled mean and sem. Useful when conditions are "Control" and "Injected" for example. If `None`, this is not plotted.
   - `PLOT_TRIALS`: `True` or `False`. Whether to plot individual trials. When there are a lot, it's a good idea not to show them to not clutter the graphs.
   - `PLOT_CONDITION`: `True` or `False`. Whether to plot mean and sem per condition.
   - `PLOT_ANIMALS`: `True` or `False`. Whether to plot mean and sem per animal.
   - `PLOT_ANIMALS_MONOCHROME`: `True` or `False`. Whether to plot each animals in the same color.
   - `PLOT_CONDITION_OFF`: `None` or a list. List of conditions NOT to be plotted. Does not apply for the raster plots nor the bar plots next to the time series.
4. Check the configs/xxx.py, where "xxx" corresponds to `MODALITY` specified in the main script. In particular, check that `PIXEL_SIZE`, `STIM_TIME` and `CLIP_DURATION` correspond to your experimental conditions if a settings.toml file is not provided. Also check that `BODYPARTS` contains all the bodyparts used to compute features. Check that the thresholds defined in the `--- Data cleaning parameters` section make sense.
5. Fill the `--- Call ---` section at the bottom of the main script. This is where the main function is called when executing the file with Python. This is where you specify the input directory that contains all the DLC files. Optionally, you can specify an output directory so that output files and figures are saved. The details of those files are given in the [Outputs](#outputs) section.

#### The configuration file
This file actually defines how are computed the requested features. It is a python file that needs to be located under `configs/xxx.py` subdirectory, relative to the `features_from_dlc.py` file. In case you want to write your own, you should start from an existing, working one (such as bundled "openfield.py"). If you have no idea where to start, ask me (Guillaume, g.legoc@posteo.org).

Note that those files are very specific : they use DLC bodyparts directly so the latter need to exist in the tracking file (eg. bodyparts should be named _exactly_ the same and correspond to _exactly_  the same bodypart). All the variables in here are used in the main script, so they all need to be defined, while imports will depend on what is needed. If you don't want to actually use them, you need to find a trick so that the functionnality is disabled but the script is still working.

This file consists in some variables defined at the top of the file, and a Python Class defining a "Config" object. This object has some required attributes and methods. Below is the extensive description of all variables.

##### Global variables
###### Physical parameters
This is where we convert images to real-world units. Those values are used only if they are missing from the settings.toml file placed next to the DLC files to analyze or if it does not exist.
- `PIXEL_SIZE`: scalar. Conversion factor from image coordinates to real-world length, expressed in unit of length (eg. mm). To measure that, you can open a video in Fiji (File > Import > Movie (FFMPEG)). Then with the Line tool, measure in pixels the length of known object. Then, `PIXEL_SIZE` will be $length_{mm}/length_{pixels}$, optionally averaging two conversion in the horizontal and vertical directions.
- `CLIP_DURATION`: scalar. Duration of the video clips, in units of time (eg. seconds). Each DLC file contains $n_ {frames}$ lines, so we use that to convert frames to real-world time.
- `STIM_TIME`: tuple with 2 elements. The onset and offset of the stimulation in the video clip, in the same units as `CLIP_DURATION`.
- `FRAMERATE`: scalar. This is **not** the actual framerate of the videos. It is used to create a generic time vector on which the time series are aligned, so that we can properly average each trace at the same time points. It should correspond to the lowest framerate you have in your videos.

###### Features
This is where we define what it is actually computed.
- `SHIFT_TIME`: bool. Display only. This is to shift all times so that the stim onset is at time 0 in the graphs.
- `BODYPARTS`: list of strings. Bodyparts of the DLC files that are actually used in the analysis. Anything that is not in there will be discarded, so it needs to contain all bodyparts used to compute features.
- `FEATURES_NORM`: tuple of 'features'. Elements in here will be normalized by subtracting their pre-stimulation mean.
- `NSTD`: scalar. The main script attempts to estimate the delay between the stimulation onset and the actual change in behaviour. This parameter controls how is defined the "change in behaviour". It is defined as when the value becomes higher than NSTD times the standard deviation of the signal before the stimulation onset.

###### Data cleaning
This is where we handle tracking errors. DLC files come with a 'likelihood' for each bodypart in each frame, ranging from 0 to 1. The latter means the model is 100% sure the bodypart was correctly assigned, the former means the opposite. If we keep low-likelihood values, we'll end up with artifacts in the computed features, so we need to get rid of them. If there are not too much, missing data can be interpolated to reconstruct a legit time serie, in some extent. Otherwise, the trial need to be discarded.
- `LH_THRESH`: scalar between 0 and 1. Likelihood below this will be considered as missing data.
- `LH_PERCENT`: scalar between 0 and 1. If a trace has more than this fraction of missing data, the whole trial is dropped and not used in the analysis.
- `LH_CONSECUTIVE`: integer. If a trace has more than this amount of consecutive frames with missing data, the whole trial is dropped and not used in the analysis.
- `INTERP_METHOD`: string, corresponding to the `method` parameter of [`pandas.DataFrame.interpolate`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html#pandas.DataFrame.interpolate). Typically "linear" or "cubic", defines the order of the interpolation method.

###### Display
Those are graphs options, such as human-readable labels and so on.
- `XLIM`: 2-elements list or None or empty list. Controls the x limits on the time series plot.
- `XLABEL_LINE`: string. x axis label for time series plots.
- `FEATURES_LABELS`: dict mapping a 'feature' to a name shown in the graphs.
- `FEATURES_YLIM`: dict mapping a 'feature' to a y-axis range. The latter should be a 2-element list `[ymin, ymax]`. To make the graphs auto-adjust their axes, use an empty dictionary `{}`.

##### The Config class
This is the object that will be instantiated in the main script. It reads the global variables defined above to make them available in the script, but especially defines the actual features and metrics computation. Some of its methods (function attached to the object) must exist. Below is the description of each components of that class.
The required methods are :
- \_\_init()\_\_
- read_setting()
- get_pixel_size()
- setup_time()
- get_features()
- write_parameters_file()
- preprocess_df()

###### \_\_init()\_\_
This function is executed when one creates a Config object. It takes as argument a settings.toml file to fill the timings attribute. If no settings.toml file is passed, the timings are read from the global variables.
###### read_setting()
This just reads a value corresponding to a key in the `settings` dictionary, if the key is missing, return the fallback value.
###### setup_time()
This creates the common time vector on which all trials are aligned to be averaged properly. If time-shifting is enabled, it takes care of that too.
###### get_pixel_size()
This is where the pixel size is read from the settings file. It is updated for each animal so that one can provide different pixel sizes for each animals. If the animal does not have an entry in the settings.toml file, the default one is used. If it does not exist either or no settings.toml file was used, the global variable is used instead.
###### get_features()
This is the main piece, where the actual computations are defined !
- `features`: dict mapping a key to a function. This is where the features computation actually happens. It maps a 'feature' to a function. It is in the form `{feature: fun(df)}`. 'feature' is an arbitrary name that will end up in the summary CSV generated at the end of the analysis, so it should be somewhat explicit. It will also be used to link each feature to other parameters (metrics, ... -- see below).
  The function 'fun' should take as input a pandas DataFrame as returned when reading a DLC file with pandas and return a 1D vector with feature values. 'fun' can be directly defined in the dict with lambda function (as in "jaw.py"), or with functions defined at the bottom of the configuration file (as in "openfield.py").
  - if you need a bodypart directly, you can simply map it to the feature : `{"feature_name": lambda df: df["bodypart", "x"]}`. 
  - if you have simple calculation to do, you can directly define it in the dictionary : `{"feature_name": lambda df: df["bodypart1", "x"] - df["bodypart2", "x"]}`.
  - if you have more complex operations to do, for instance that require intermediate values to be computed, you can wrap it up in a function defined afterwards : `{"feature_name": lambda df: self.my_fun(df)}`. Then you need to define `my_fun` as a method. `my_fun` can in turn call any other user-defined functions, as long as it returns a 1D vector in the end.
  
In the following variables, each feature is referred to by its key in the `features` dictionary.

- `features_metrics`: dict mapping a 'feature' to another dict. The latter maps a metric name to a function that returns a scalar from a 1D vector. This is the metric used to quantify the change during the stimulation. The operations will be applied to each 'features' during the time range specified in `features_metrics_range` (see below). It should return a scalar, so we have a scalar associated to each trial. Those scalars are then represented as bar plots next to the time series. Each feature can have any number of metrics to show on the side.
- `features_metric_range`: same structure as `features_metrics`, but mapping to 2-elements lists defininig the time range in which the metric defined above is computed. The units should be the same as `clip_duration`, and expressed _before_ time shifting (see `SHIFT_TIME`), eg. timings as in the original video clips.
- `features_metrics_share`: same structure as `features_metrics`, but mapping to a boolean (True/False). It specifies whether the metric should share its y-axis with the feature time serie (eg. the feature and the metric are in the same units and range).
- `features_labels` : just reads the `FEATURES_LABELS` global variable, specify the name displayed on the y axis of the features time series.

###### write_parameters_file()
This just takes a bunch of the configuration parameters and write them to a text file for reference.

###### preprocess_df()
It should take as input a DataFrame and return the same DataFrame (with same columns and index). This is used to mark some time steps as missing, when custom criteria are met, independently of the likelihood. This exists because DLC can confidently place markers in wrong places. It can be specific to your data, eg. have hardcoded DLC bodyparts. Practically, the function should scan the DataFrame, spotting weird values and replace them with `np.nan`. Then, the main script will consider it as missing and the values will be interpolated, if able (based on dropping criteria, see the [Data cleaning section](#data-cleaning)).

#### The settings.toml file
A template file is provided in the `resources` folder. It can be copied next to the DLC files to be analysed to specify the experiments parameters, including :
- `clip_duration` : duration of video clips in time units (typically seconds).
- `stim_time` : two-elements list with onset and offset times of the stimulation (in the same units as `clip_duration`).
- `framerate ` : framerate of the common time vector used to align all time series.

Then, you have a `[default]` section, which (for now) contains only the pixel size.
Finally, you can have any number of sections corresponding to animals ID, containing, as well, only the pixel size. This allows one to specify a different pixel size for each animal. The logic is the following :
- If a settings.toml file is found next to the files to be analyzed, then the timings values are read from there and used for the whole analysis. If a field is missing, it fallbacks to the global variables in the configuration file.
- For a given animal, pixel size is read from its section in the settings.toml file. If the animal does not have a section, the value in the `[default]` is used. If it does not exist either, the global variable in the configuration file is used.
- If no settings.toml file is found during analysis, the global variables in the configuration file are used.

#### The plot configuration file
This is a TOML file specifying various display options. Generated graphs use it to set their line colors, axes color and thickness, style, etc. Everything could be modified in post-production in a vector graphics editor software such as [Inkscape](https://inkscape.org/), but this file can allow you to spend a whole lot less time editing your graphs.

As a TOML file, it has sections declared by brackets (`[section name]`), followed by entries declared with an equal (`something = anything`). Most python types are supported, eg. strings (`"strings"`), scalars (`5.25`, `8`), booleans (`true`, `false`) and lists (`[0.7961, 0.9451, 0.9961]`). The file is extensively documented, which means each lines is explained with comments (`# comment`).

Basically, this controls line colors, transparency, width for each type of trace (per-trial, per-animal, per-condition and pooled). It further specifies the properties of the stimulation representation (color, transparency and name in the legend). It also controls the bar plots properties (bars face and edges colors, and error bar cap size). Finally, it sets the axes properties (the x and y axes thickness, number of ticks, font family and size, whether to display the grid and arrows at the tip of axes and figure size).

For line colors : sometimes it's unique (eg. for trials that are all plotted in the same colors, or the pooled average which is unique), so the color is a three-element list giving the R, G, B values of the line. For others, it requires a color per elements that are plotted. For instance, if I have 2 conditions, the conditions color must be a list of 2 3-elements list, for instance `color=[[0.7961, 0.9451, 0.9961], [0.15, 0.2544, 0.7660]]`. The first condition will be associated to the first RGB color, and so on. When we don't know the number of elements that will be plotted, or we don't want to manually select the colors, we can use colormaps or palette. They are referred to with a name and all matplotlib's colormaps are available in its [documentation](https://matplotlib.org/stable/users/explain/colors/colormaps.html#sequential). One should then specify the color or palette as `color="name_of_colormap"`.

#### Outputs
Upon completion, several files are saved if an output directory was specified. Note that if files with the same names previously existed in this directory, they will be overwritten without warning.

Each generated figures are saved as svg (editable vector graphics). They are saved as you see them on the screen, so if for instance the figures are too small and the text are overlapping, it will propagate to the svg files. You can optionally resize the figure and save it again from the figure's graphical interface, or modify the options in the [plot configuration file](#6. The plot configuration file).

Along with the figures, you'll find two log files : `dropped.txt` and `used.txt`. The former contains the files that were dropped because there were too much missing data (based on the likelihood criteria set in the configuration file, see the [Data cleaning section](#data-cleaning)). Conversely, the `used.txt` contains the files that were used in the analysis. This excludes the files from `dropped.txt` along with the ones that were not assigned to a condition. In that case, a warning message is issued in the console.

A summary CSV is saved. It contains each features' time series for each trials, giving the trial number, the trial ID, the condition, the filename and the animal.

Last, a parameters file tracks the parameters used for this analysis.

## Vocabulary
- [python](https://www.python.org/) is a popular programming language that is free and open-source. It can run on mostly any hardware with any operating-system. Its capability on its own are limited, but there are hundreds of community-driven libraries that extend its functionality and make it easy to handle data and visualization. It makes it harder to get started because you need to install those libraries, unlike MATLAB that is a all-in-one solution.
- markdown (.md, such as this file) is a specific formatting syntax that makes text file nice to write and read. It can be opened with any text editor (such as notepad) but you can see it rendered with Visual Studio Code (top right magnyfing glass).
- toml (.toml) is a specific formatting syntax that makes text files human- and computer- readable which is useful for configuration files (such as config_plot.toml and settings.toml).

Python-specific vocabulary :
- library : bunch of modules and functions you call in your script. There are libraries bundled with bare python (standard library), and others you need to install
- environment : because of those libraries, it can quickly becomes a mess. Some libraries depend on others, and all have different versions that in turn require specific version. So installing all libraries in one place will probably lead to dependency issues (libA 1.2.1 requires libB 1.3 which requires libC 2.5.7, but libD 3.5.4 requires libB 1.2.1). So what we do is use "environment", where all required libraries in their specific version are installed in a dedicated space and independent on other environments. They are handled by an package and environment such as conda.
- [conda](https://docs.anaconda.com/free/miniconda/index.html) is a widely used environment and package manager. It is bundled with Anaconda.
- [Anaconda](https://www.anaconda.com/download/) is an attempt to make it easy to use python (an all-in-one solution like MATLAB). But in my opinion it's quite broken and ultimately it is way easier and faster to use the conda only, with a few command lines in a terminal.
- [types](https://docs.python.org/3/library/stdtypes.html) : these are the kind of variables python handles, such a float, integer, string, boolean, list and dictionary. Functions expect a specific types and will raise an error if the wrong type is provided. So you need to be careful in what you provide. Short list :
  - boolean : `True` or `False`.
  - int : integer. 1, 2, 1e5, ...
  - float : float numbers. 1.5, 2.1, -5.8784521, 10/3, ...
  - list : `["a", 5, "hello", True]`. Enclosed with `[]`, even if it's one element.
  - tuple : `("a", 5, "hello", True)`. Enclosed with `()`, somewhat similar to list (but not exactly). For single element tuple, it's like this : `("a", )`.
  - dictionary : `{"key": 5, "something": [1, 2, 3]`. Links a key to a value, the latter can be any python types.
  - None : python nothing.
- docstrings : enclosed by triple quotes (`"""docstring"""`), they provide useful informations on what function does and what it expects as inputs.

## Tips
- if there is an error, read the error message. Often, it might indicate what went wrong (for instance, could not locate a file, no data, etc.).
- google (or any search engine) is your best friend when it comes to programs and code. Copy/pasting error messages might give you answers.