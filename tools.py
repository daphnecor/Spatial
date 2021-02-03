'''
Functions in toolbox to generate various plots. From:
https://github.com/mattperich/TrialData/tree/master/Tools
'''
import numpy as np


def remove_bad_trials(trial_data, ranges, remove_nan_idx=False, nan_idx_names, verbose=False):
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
                    # ... More sanity checks 

    # IF index is bad --> 1
    bad_idx[:, trial] = 1

if verbose:
    print(f'Removed {sum(bad_idx)} trials')

bad_trials = trial_data[np.where(bad_idx==1)]
trial_data = trial_data[np.where(bad_idx==0)]

return bad_trials, trial_data


def remove_bad_neurons(trial_data, ranges, remove_nan_idx=False, nan_idx_names, do_shunt_check=False, prctile_cutoff=99.5, do_fr_check=True,
                      min_fr=0, fr_window=[], calc_fr=False, verbose=False):
    """"
    Checks for shunts or duplicate neurons based on coincidence, and also removes low-firing cells.
    LINK: https://github.com/mattperich/TrialData/blob/master/Tools/removeBadNeurons.m

    Parameters
    ----------
    trial_data: pd.DataFrame
        data in trial_data format

    ranges: dict
        {'idx_START','idx_END',[MIN_#BINS,MAX_#BINS]...} 
        ex: {'idx_go_cue','idx_movement_on',[5 30]} to remove
          reaction times smaller than 5 and larger than 30 bins
    remove_nan_idx: bool (optional, default False)
        removes trials any idx with NaN values.
    nan_idx_names: str of list of str
        which fields for remove_nan_idx. Default is to do 'all'
    do_shunt_check: bool (optional, default False)
        ...
    prctile_cutoff: float
        cutoff threshold in percentage
    do_fr_check: bool (optional, default True)
        ...
    min_fr: float
        ...
    fr_window: ...
        ...
    calc_window: bool (optional, default False)
        ...
    use_trials: np.array
        ...
    
    Returns
    -------
    trial_data: the dataframe with bad units removed
    
    bad_units: list of indices in the original dataframe thata are bad
    """





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








