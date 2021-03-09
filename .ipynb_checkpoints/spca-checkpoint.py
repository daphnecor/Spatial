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