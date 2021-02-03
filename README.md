# Folders

- 📦  `pyaldata` contains the existing code base (created by ...)
- 📊  `explore.ipyb` is used to plot figures
- 🔧 `tools.py` are the functions used to plot figures, preprocess data etc. based on the [existing matlab repo Trialdata](https://github.com/mattperich/TrialData)

# General workflow

### Part 0 - Data structure

- Load data and convert to pandas 

### Part 1 - Preprocess data

- Select data we want (`@getTDidx`)
- Remove Bad Trials (`@removeBadTrials`) $\rightarrow$ [removeBadTrials](https://github.com/mattperich/TrialData/blob/master/Tools/removeBadTrials.m) 
- Remove Bad Neurons (`@removeBadNeurons`) $\rightarrow$ [removeBadNeurons.m](https://github.com/mattperich/TrialData/blob/master/Tools/removeBadNeurons.m)
    - Get TD fields $\rightarrow$ [getTDfields.m](https://github.com/mattperich/TrialData/blob/master/Tools/getTDfields.m)
- Transform (sqrt) (`@srqtTransform`)
- Smoothen signals (`@smoothSignals`)
- Trim the TD (`@trimTD`)

### Part 2 - Dimensionality reduction

- Choose dimensionality reduction method
    - PCA (`@getPCA`)
    - ...


### Part 3 - Plot leading PC Near vs Far 

- same electrode
- same array
- other array


## Mapping

![](https://raw.githubusercontent.com/daphnecor/Pyaldata/master/Overview.jpg)
