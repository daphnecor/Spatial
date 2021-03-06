"""

Dependencies

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from sklearn import decomposition
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


def getPCA(rates):
	"""
	Performs PCA on firing rates
	
	Parameters
	----------
	rates:
		
	Returns
	-------
	loadings: the eigenvectors
		Each element represents a loading, namely how much (the weight) each 
		original neuron contributes to the corresponding principal component
	explained_var: ratio of explained variance
	""" 
	X = np.concatenate(rates.values, axis=0)
	pca = decomposition.PCA()
	X = pca.fit_transform(X)

	return pca.components_, pca.explained_variance_ratio_


def compare_pc_weights(m1_arr, m1_ug, pmd_arr, pmd_ug, w):
	"""
	Compares the PC weights across distance
	
	Parameters
	----------
	m1_arr: np.array
	   electrode map containing the spatial location of the m1 electrodes 
	
	m1_ug: 
		unit guide of the m1 array
	
	pmd_arr: np.array
		electrode map containing the spatial location of the m1 electrodes

	pmd_ug: 
		unit guide of the pmd array
	
	w: np.array 
		vector with the weights or loadings of the first principal component (all neurons)
	
	Returns
	-------
	df: pandas dataframe 
		...

	Note that this assumes that the weight vector is constructed such that all M1 weights are first, 
	and all PMd weights are second like [M1, M1, ... , PMd , PMd]
	"""

	# compare within m1 array
	within_m1_dist, within_m1_w = [], [] 

	for i in range(len(m1_ug[:, 0])): # loop along neurons 
		# find electrode that corresponds to this neuron
		elec1 = m1_ug[i, 0]
		loc1 = np.where(m1_arr == elec1) # find neuron location on array 

		for j in range(i+1, len(m1_ug[:, 0])): # compare to all other electrodes (j!=i)        
			# find electrode location of this neuron within same array
			elec2 = m1_ug[j, 0]
			loc2 = np.where(m1_arr == elec2)

			# find euclidean distance between two neurons on array
			dst = distance.euclidean(loc1, loc2)

			within_m1_dist.append(dst) 
			within_m1_w.append(np.abs(w[j] - w[i])) 

	# compare within pmd array
	within_pmd_dist, within_pmd_w = [], [] 

	for i in range(len(pmd_ug[:, 0])): # loop along neurons 
		# find electrode that corresponds to this neuron
		elec1 = pmd_ug[i, 0]
		loc1 = np.where(pmd_arr == elec1) # find neuron location on array 

		for j in range(i+1, len(pmd_ug[:, 0])): # compare to all other electrodes (j!=i)        
			# find electrode location of this neuron within same array
			elec2 = pmd_ug[j, 0]
			loc2 = np.where(pmd_arr == elec2)

			# find euclidean distance between two neurons on array
			dst = distance.euclidean(loc1, loc2)

			within_pmd_dist.append(dst) 
			within_pmd_w.append(np.abs(w[j] - w[i])) 
	
	# compare pmd to m1
	pmd_m1_w = []
	
	for i in range(len(pmd_ug[:, 0])): # loop along neurons 
		# compare to all neurons in other array
		for j in range(len(m1_ug[:, 0])):
			# compare all weights from main to all weights from other
			pmd_m1_w.append(np.abs(w[pmd_ug.shape[0]+j] - w[i]))

	df = pd.DataFrame({'distance':np.concatenate((within_m1_dist, within_pmd_dist, np.full(len(pmd_m1_w), np.nan)), axis=0), 
		'w_diff': np.concatenate((within_m1_w, within_pmd_w, pmd_m1_w), axis=0)})

	# group them to make plotting easier
	df['array'] = np.nan
	df['array'].iloc[:len(within_m1_w)] = 'M1'
	df['array'].iloc[len(within_m1_w): len(within_m1_w) + len(within_pmd_w)] = 'PMd'
	df['group'] = df['distance'].apply(lambda d: 'same elec' if d == 0 else ('same array' if d > 0 else ('other array')))

	return df
	

def display_grid(arr):
	''' Displays the number of neurons per electrode on array.'''
	
	fig, ax = plt.subplots(1, figsize=(5,5), dpi=80)
	ax.imshow(arr, cmap='Greys')
	for i in range(len(arr)):
		for j in range(len(arr)):
			text = ax.text(j, i, int(arr[i, j]), ha='center', va='center', color='w')



def sort_pcs_by(pcs, by, abs_val=True):
    '''
    Sort the principal components loadings by the indices of the dimension 'by' in descending order.
    
    Parameters
    ----------
    pcs: np.array
        N x N matrix with the principal components
    by: number of the pc by which you want to sort the others
    
    Returns
    -------
    M: np.array
    '''

    if abs_val:
        pcs = abs(pcs)
    else:
        pcs = pcs
    
    # get the weights of the first n neurons --> M1 & PMd neurons
    pcs_m1  = pcs[0:td['M1_spikes'][0].shape[1], :]
    pcs_pmd = pcs[td['M1_spikes'][0].shape[1]:, :]

    # sort 
    W_m = pcs_m1[np.argsort(-pcs_m1[:, by]), :]
    W_p = pcs_pmd[np.argsort(-pcs_pmd[:, by]), :]

    return np.concatenate((W_m, np.full((1, W_m.shape[1]), np.nan), W_p), axis=0)



class SparsePCA():
	"""
	Sparse principal component analysis
	"""

	def __init__(lam, tol=1e-8, max_iter=100, verbose=False):
		self.lam = lam
		self.tol = tol
		self.max_iter = max_iter
		self.verbose = verbose

	def fit(self, X):
		"""Fit the model from data in X

		Parameters
		----------
		

		Returns
		-------

		"""
		
		# demean data
		self.mean_ = X.mean(axis=0)
		X = X - self.mean_

	def learn():
		"""
		"""
		pass








# def get_neuron_elec_mapping(elec_map, unit_guide, spikes, trial):
#     '''
	
#     Parameters
#     ----------
#     elec_map:
	
#     unit_guide: 
	
#     Returns
#     -------
#     unit_arr: 

#     '''
#     elecs = list(range(1, 97)) # total number of electrodes
#     unit_arr = np.zeros((10, 10)) # assuming 10x10 array
#     neuron_distrib = []
#     elecs_to_spikes = {}
	
#     # get the spike trains from a particular trial
#     spikes_trial_i = spikes[trial]

#     for e in elecs:

#         # find indices in unit guide of this electrode number 
#         e_indices = np.where(unit_guide[:, 0] == e)

#         if np.size(e_indices) == 0: # skip if electrode number is not in col
#             neuron_distrib.append(0)
#             elecs_to_spikes.update({e: np.array([])}) # add empty array?
#             continue

#         # take neurons that belong to these indices 
#         neurons_at_e = max(unit_guide[:, 1][e_indices])

#         neuron_distrib.append(neurons_at_e) # largest number is total number of neurons

#         arr_idx = np.where(elec_map == e) # get spatial location of electrode
#         unit_arr[arr_idx] = neurons_at_e # store number of neurons in array

#         # spiketrains of neurons that belong to electrode e
#         spikes_elec_k = np.array([spikes_trial_i[:, k] for k in e_indices])[0]

#         # append to dictionary with electrode num as key and spiketrains as values
#         elecs_to_spikes.update({e:spikes_elec_k})
		
#     return neuron_distrib, unit_arr, elecs_to_spikes
























