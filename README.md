# Folders

- 📦  `pyaldata` contains the utils and tools scripts from the 'real' [PyalData repo](https://github.com/mattperich/PyalData) 
- 📊  `main.ipyb` is used to plot figures and generate results
- 🌲  `explore.ipynb` is just to try things out, get to know the data, things like that
- 🔧 `tools.py` are the functions used to plot figures, preprocess data etc. based on the [existing matlab repo Trialdata](https://github.com/mattperich/TrialData)

# General workflow

### Part 0 - Data structure

- Load data and convert to pandas 

### Part 1 - Preprocess data

- Select data we want (`@getTDidx`)
- Remove Bad Trials (`@removeBadTrials`) $\rightarrow$ [removeBadTrials](https://github.com/mattperich/TrialData/blob/master/Tools/removeBadTrials.m) 
- Remove Bad Neurons (`@removeBadNeurons`) $\rightarrow$ [removeBadNeurons.m](https://github.com/mattperich/TrialData/blob/master/Tools/removeBadNeurons.m)
    - Get TD fields $\rightarrow$ [getTDfields.m](https://github.com/mattperich/TrialData/blob/master/Tools/getTDfields.m)
- Transform (sqrt) (`@srqtTransform`) $\rightarrow$ [sqrtTransform.m
](https://github.com/mattperich/TrialData/blob/master/Tools/sqrtTransform.m)
- Smoothen signals (`@smoothSignals`) $\rightarrow$ [smoothSignals.m](https://github.com/mattperich/TrialData/blob/master/Tools/smoothSignals.m)
- Trim the TD (`@trimTD`) $\rightarrow$ []()

### Part 2 - Dimensionality reduction

- Choose dimensionality reduction method
    - PCA (`@getPCA`)
    - FA
    - ...


### Part 3 - Find things out


## Mapping [TODO]

![](https://raw.githubusercontent.com/daphnecor/Pyaldata/master/Overview.jpg)
