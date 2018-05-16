# Algoritmo CHAT-OHM creado por Elias Ventura-Molina

import numpy as np

class CHAT_ONEHOT:
	def __init__( self, ds ):
		self.ds = ds
		self.mean = np.array( [] )
		self.mask = []

	def train( self ):
		m = np.size( self.ds, axis=0 )  # np.unique( self.ds[:, -1] ).size
		n = np.size( self.ds, axis=1 ) - 1
		self.mean = np.mean( self.ds[:, 0:-1], axis=0 )
		self.M = np.zeros( (m, n) )
		for i, p in enumerate( self.ds ):
			xM = np.subtract( p[0:-1], self.mean )
			yM = np.array( [int( x == i ) for x in range( m )] )
			self.M = np.add( self.M, np.outer( yM, xM ) )
			self.mask += [p[-1]]
		return self.M

	def classify( self, p ):
		xM = np.subtract( p, self.mean )
		yM = np.inner( self.M, xM )
		maximum = max(yM)
		#print [np.sum([self.mask[_] == i and x == maximum for _, x in enumerate( yM )]) for i in range( np.unique( self.ds[:, -1] ).size )]
		classes = [np.sum( [self.mask[_] == i and x == maximum for _, x in enumerate( yM )] ) for i in range( np.unique( self.ds[:, -1] ).size )]

		return np.argmax( classes )
