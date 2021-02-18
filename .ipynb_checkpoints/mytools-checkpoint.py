'''
Functions in toolbox to generate various plots. From:
https://github.com/mattperich/TrialData/tree/master/Tools
'''
import numpy as np
import pandas as pd


def remove_cmp_formatting(s):
    """
    Used in read_cmp() to remove formatting in .cmp file
    
    Parameters
    ----------
    s: str
        one line in the file
        
    Returns
    -------
    list of strings
    """
    for r in (('\t', ' '), ('\n', ''), ('elec', '')):
        s = s.replace(*r)       
    return s.split() 


def read_cmp(file_path):
    """
    Read in Blackrock Microsystems .cmp file into Python
    
    Parameters
    ----------
    file_path: str
        .cmp file path + name 
        
    Returns
    -------
    df_array: dataframe of shape (num electrodes, 5)
        [col (int), row (int), channel number (str), within_channel_num (int), global electrode number (int)]
    """
    # Open file, remove comments and remove other formatting we don't need
    with open(file_path) as f:
        temp = [line for line in f if not line.startswith('//')]     
    clean_lsts = [remove_cmp_formatting(l) for l in temp[1:]]

    df = pd.DataFrame(clean_lsts, columns=['array_col', 'array_row', 'channel_num', 'within_channel_num', 'global_enum']).dropna()
    
    # Convert columns to integers - errors='igore' return the column unchanged if it cannot be converted to a numeric type
    df_array = df.apply(pd.to_numeric, errors='ignore')
    
    return df_array


def remove_bad_trials(trial_data, ranges, nan_idx_names, remove_nan_idx=False, verbose=False):
    """"
    1) Will remove any trials with any idx_ fields that are NaN
    2) Will remove trials according to ranges input

    LINK: https://github.com/mattperich/TrialData/blob/master/Tools/removeBadTrials.m

    Parameters
    ----------
    trial_data: pd.DataFrame
        data in trial_data format

    ranges: dict
        {'idx_START','idx_END',[MIN_#BINS,MAX_#BINS]...}: {'idx_go_cue','idx_movement_on',[5 30]} 
        to remove reaction times smaller than 5 and larger than 30 bins

    remove_nan_idx: bool (optional, default False)
        removes trials any idx with NaN values.

    nan_idx_names: str or list of str
        which fields for remove_nan_idx. Default is to do 'all'
    
    Returns
    -------
    trial_data: the dataframe with bad trials removed

    bad_units: the dataframe containing said bad trials
    """

    use_trials = np.arange(0, len(trial_data))
    
    # TODO: ln 39-40 
    # trial_data = check_td_quality(trial_data);
    # if ~iscell(nan_idx_names), nan_idx_names = {nan_idx_names}; end

    fn_time = getTDfields(trial_data, type='time');

    # TODO: ln 44-47

    # construct empty array. If idx is bad: 1, if idx is ok: 0.
    bad_idx = np.zeros([1, len(trial_data)])

    # iterate through trials (rows wise) in one session
    for idx, td in trial_data.iterrows():

        if remove_nan_idx: # loop along all indices and make sure they aren't NaN (remove NaNs)
            td = td.dropna() # TODO: dropna specifics (ln 63)
        
        # TODO: isfield() ln 68-81
        
        # Look for trials that are outside the allowable length
        if len(ranges) != 0: # if ranges is not empty
            if ranges.shape[1] != 3: assert('Ranges input not properly formatted.')

            for i in range(ranges.shape[0]):
                # define index values so I can check to make sure it's okay

                # TODO: deal([]) ln 89
                if ranges[i, 0] == 'start': idx1 = 0
                elif ranges[i, 1] == 'end': # TODO


                    # if the requested values don't exist 
                    if len(idx1) == 0:
                        # TODO
                        assert('idx references are outside trial range.')
                    elif len(idx2) == 0:
                        # TODO
                        assert('idx references are outside trial range.')

                        # ASK: goal of this, can this be done more efficiently?    
                        # ... More sanity checks follow

    # IF index is bad --> 1
    bad_idx[:, trial] = 1

    if verbose:
        print(f'Removed {sum(bad_idx)} trials')

    bad_trials = trial_data[np.where(bad_idx==1)]
    trial_data = trial_data[np.where(bad_idx==0)]

    return bad_trials, trial_data


def remove_bad_neurons(trial_data, do_shunt_check=False, prctile_cutoff=99.5, do_fr_check=True,
                      min_fr=0, fr_window=[], calc_fr=False, verbose=False):
    """"
    Checks for shunts or duplicate neurons based on coincidence, and also removes low-firing cells.
    LINK: https://github.com/mattperich/TrialData/blob/master/Tools/removeBadNeurons.m

    Parameters
    ----------
    trial_data: pd.DataFrame
        data in trial_data format

    remove_nan_idx: bool (optional, default False)
        removes trials any idx with NaN values.

    nan_idx_names: str of list of str
        which fields for remove_nan_idx. Default is to do 'all'

    do_shunt_check: bool (optional, default False)
        flag to look for coincident spiking

    prctile_cutoff: int
        value (0-100) for empirical test distribution

    do_fr_check: bool (optional, default True)
        flag to look for minimum firing rate

    min_fr: float
        minimum firing rate value to be a good cell 
        NOTE: assumes it's already a firing rate, i.e., it won't divide by bin size

    fr_window: ... #TODO
         when during trials to evaluate firing rate {'idx_BEGIN',BINS_AFTER;'idx_END',BINS_AFTER}
    
    calc_fr: bool (optional, default False)
        will divide by bin_size if true
    
    use_trials: np.array
        can only use a subset of trials if desired
    
    Returns
    -------
    trial_data: the dataframe with bad units removed
    
    bad_units: list of indices in the original dataframe that are bad
    """

    # ASK: check_td_quality(trial_data) 

    # determine bin size, all are the same so we can take any trial
    bin_size = trial_data.iloc[0, :]['bin_size']

    # select subset of the data if preferred, otherwise use all data
    #if len(use_trials) != len()
        #use_trials = getTDidx(trial_data, col, v); # Existing function in Pyaldata

    # TODO ...

    # ASK: there is a function in Pyaldata called 'remove_low_firing_neurons' but this 
    # does not check for shunts based on coincidence.



def getTDfields(trial_data, type): # TODO: build this function
    """
    Will return a cell array which lists all of the field names of a given type. 
    Useful to list all spiking variables, kinematic variables, etc.

    Parameters
    ----------
    trial_data: pd.DataFrame
        data in trial_data format
    which_type: str
        the type of field. Options:
           1) 'cont'        : continuous (pos, vel, force, etc)
           2) 'spikes'      : neural data fields
           3) 'arrays'      : similar to spikes but returns array names
           4) 'time'        : names of all time varying fields
           5) 'idx'         : name of all time index fields
           6) 'neural'      : any neural signals (e.g. M1_WHATEVER)
           7) 'unit_guides' : all unit_guides
           8) 'labels'      : for naming signals (unit_guides or _names)
           9) 'meta'        : all fields that are not time-varying or idx_

    Returns
    -------
    fn: ...
        the fieldnames of which_type
    """

    pass


def sqrtTransform():
    """
    Already exists
    """
    pass










