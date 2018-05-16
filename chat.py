#Algoritmo CHAT original creado por Elias Ventura-Molina

import numpy as np


class CHAT:
	def __init__( self, ds ):
		self.ds = ds
		self.mean = np.array( [] )

	def train( self ):
		m = np.unique( self.ds[:, -1] ).size
		n = np.size( self.ds, axis=1 ) - 1
		self.mean = np.mean( self.ds[:, 0:-1], axis=0 )
		self.M = np.zeros( (m, n) )
		for p in self.ds:
			xM = np.subtract( p[0:-1], self.mean )
			yM = np.array( [int( i == p[-1] ) for i in range( m )] )
			self.M = np.add( self.M, np.outer( yM, xM ) )
		return self.M

	def classify( self, p ):
		xM = np.subtract( p, self.mean )
		yM = np.inner( self.M, xM )
		return np.argmax( yM )
