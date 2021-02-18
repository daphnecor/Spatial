"""

Dependencies

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)


"""

Functions

"""

def make_raster(spike_trains):
    '''
    Plots a raster given a set of (binary) spike trains.
    
    Parameters
    ----------
    spike_trains: np.array 
        set of binary arrays in the form [# timepoints, # neurons]
    '''
    if np.size(spike_trains) == 0:
        print('This array is empty')
        return
    
    # In the case some entries > 1 convert them to 1
    binary_trains = np.isin(spike_trains, range(1, 10)).astype(np.uint8) 
    
    # Scale length of plot by number of neurons   
    if spike_trains.shape[1] < 2: 
        flen = 1
    else: 
        flen = spike_trains.shape[1]/5
    
    fig, ax = plt.subplots(1, figsize=(12, flen), dpi=80)
    for i in range(spike_trains.shape[1]): 
        y_val = i + 1 
        spike_train_i = binary_trains[:, i] * y_val
        spike_train_i = [float('nan') if x==0 else x for x in spike_train_i] 

        plt.scatter(range(spike_trains.shape[0]), spike_train_i,  marker='|', c='k', s=50);
        ax.set_title(f'Raster plot of {spike_trains.shape[1]} neuron(s)', fontsize=15)
        ax.set_xlabel('time', fontsize=13)
        ax.set_ylabel('neuron', fontsize=13)


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
    Read Blackrock Microsystems .cmp file into Python
    
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


def localize_elecs(df, elecs, N=10, verbose=False):
    """
    Get the spatial location of electrodes on the array. 
    Set verbose=True to visualise the array.

    Parameters
    ----------
    df: pd.DataFrame
        .cmp file information
    elecs: lst 
        list of electrodes for which to get location on array
    N: int 
        number of rows and cols in array

    Returns
    -------
    elec_map: np.array
        each element in the electrode map is an electrode number 
    """
    
    elec_map = np.zeros((N, N))
    
    for e in elecs:
        if e in df['global_enum'].values: 
            # find row and column of electrode 
            i = int(df.loc[df['global_enum'] == e]['array_row'])
            j = int(df.loc[df['global_enum'] == e]['array_col'])
            elec_map[i, j] = e # put electrode number at this location
        else:
            if verbose:
                print(f'Electrode number {e} does not exist in array \n')
            continue
    
    if verbose: # display array with number of neurons
        fig, ax = plt.subplots(1, figsize=(6,6), dpi=80)
        
        ax.imshow(elec_map, cmap=cmap, interpolation='none', vmin=0, vmax=1, aspect='equal')
        # code to annotate the grid and draw white squares around each cell
        def rect(pos):
            r = plt.Rectangle(pos-0.5, 1,1, facecolor='none', edgecolor='w', linewidth=2)
            plt.gca().add_patch(r)
        x,y = np.meshgrid(np.arange(elec_map.shape[1]), np.arange(elec_map.shape[0]))
        m = np.c_[x[elec_map.astype(bool)], y[elec_map.astype(bool)]]
        for pos in m: 
            rect(pos)
        for i in range(len(elec_map)):
            for j in range(len(elec_map)):
                text = ax.text(j, i, int(elec_map[i, j]), ha='center', va='center', color='w')
    return elec_map


def elecs_to_neurons(elec_map, unit_guide, elecs=list(range(1, 97)), N=10):
    """
    Get mapping between electrode number amount of neurons per electrode
    
    Parameters  
    ----------
    elec_map: np.array 
        NxN array with electrode numbers as elements

    unit_guide: np.array 
        specifies the number of neurons each electrode covers 
        note that this mapping is the same for all trials

    Returns
    -------
    cell_distrib: np.array
        distribution of neurons on given array

    cells_on_arr: np.array
        array with number of neurons for each electrode
    """

    cells_on_arr = np.zeros((N, N))
    cell_distrib = []

    for e in elecs: 
        e_indices = np.where(unit_guide[:, 0] == e)

        if np.size(e_indices) == 0: 
            cell_distrib.append(0)
        else:
            neurons_at_e = max(unit_guide[:, 1][e_indices])
            
            cell_distrib.append(neurons_at_e)
            e_loc = np.where(elec_map == e)
            cells_on_arr[e_loc] = neurons_at_e

    return cell_distrib, cells_on_arr


def elecs_to_spikes():
    """
    Get mapping between electrode numbers and spike indices in df for all trials

    Parameters  
    ----------
    elec_map: the NxN array with electrode numbers as elements.

    unit_guide:


    Returns
    -------

    """




def display_grid(arr):
    ''' Displays the number of neurons per electrode on array.'''
    
    fig, ax = plt.subplots(1, figsize=(5,5), dpi=80)
    ax.imshow(arr, cmap='Greys')
    for i in range(len(arr)):
        for j in range(len(arr)):
            text = ax.text(j, i, int(arr[i, j]), ha='center', va='center', color='w')


































