import tensorflow as tf

class DiffEq():
    """
    This class defines the different differential
    equations used.
    """

    def __init__(self, diffeq, x, y, dydx, d2ydx2):
    	"""
    	diffeq : name of the differential equation used (ex: diffeq = "first order ode")
    	"""
    	self.diffeq = diffeq
    	self.x = x
    	self.y = y
    	self.dydx = dydx
    	self.d2ydx2 = d2ydx2

    	if self.diffeq == "first order ode":
    		self.eq = self.dydx + self.y - tf.math.exp(-self.x) * tf.math.cos(self.x)

    	#hbar = 1, m = 1, omega = 1
    	#Customize the energy level with n
    	if self.diffeq == "schrodinger":
    		self.n = 1
    		self.eq = self.d2ydx2 + 2 * ( (self.n + 1/2) - tf.square(self.x)/2) * y

    	#Customize n
    	if self.diffeq == "burst":
    		self.n = 10
    		self.eq = self.d2ydx2 + (self.n**2 - 1)/tf.square(1 + tf.square(self.x)) * self.y

