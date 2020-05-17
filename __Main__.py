import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')
import matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
from matplotlib.colors import LogNorm
import scipy
from scipy import special

#Import the Dictionary class
from Dictionary import Dictionary
D = Dictionary()
Dict = D.Dict

#Import the ODE solver class
from ODEsolver import ODEsolver

#Import the DiffEq class
from DiffEq import DiffEq

#Random seed initialization
seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)



#--------------------------------------------------------------------
#-----------------FIRST-ORDER-ODE------------------------------------
#--------------------------------------------------------------------

if __name__ == "__main__":
	
	#----------------------------------
	#-----------Figure-1---------------
	#----------------------------------
	
	#Differential equation
	order = 1
	diffeq = "first order ode"
	#Training domain
	x = np.linspace(0, 2, 100)
	#Initial conditions
	initial_condition = (0, 0)
	#Number of epochs
	epochs = 10000
	#Structure of the neural net (only hidden layers)
	architecture = [10]
	#Initializer used
	initializer = Dict["initializer"]["GlorotNormal"]
	#Activation function used
	activation = Dict["activation"]["sigmoid"]
	#Optimizer used
	optimizer = Dict["optimizer"]["Adam"]
	#Save predictions at each epoch in self.predictions
	prediction_save = False
	#Save the weights at each epoch in self.weights
	weights_save = False


	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	#Training
	history = solver.train()
	epoch, loss = solver.get_loss(history)


	#Plot the exact and the neural net solution
	x_predict = x
	y_predict = solver.predict(x_predict)
	y_exact = np.exp(-x_predict)*np.sin(x_predict)
	print(solver.mean_relative_error(y_predict, y_exact))
	solver.plot_solution(x_predict, y_predict, y_exact)
	

#--------------------------------------------------------------------
#-----------------SCHRODINGER----------------------------------------
#--------------------------------------------------------------------

if __name__ == "__main__":

	#Exact solution
	def schrodinger(n, x):
		n = abs(n)
		return 1/(np.sqrt(2**n*np.math.factorial(n))) * 1/np.pi**(1/4) * np.exp(-x**2/2) * scipy.special.eval_hermite(n, x)
	
	#----------------------------------
	#-----------Figure-2---------------
	#----------------------------------
	
	order = 2
	diffeq = "schrodinger"
	n = 2
	x = np.linspace(-5, 5, 100)
	x0, y0 = 2, schrodinger(n, 2)
	dx0, dy0 = -2, schrodinger(n, -2)
	initial_condition = ((x0, y0), (dx0, dy0))
	architecture = [50]
	initializer = Dict["initializer"]["GlorotNormal"]
	activation = Dict["activation"]["sigmoid"]
	optimizer = Dict["optimizer"]["Adam"]
	prediction_save = False
	weights_save = False

	#Plotting for epochs
	epochs = 1000
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch, loss = solver.get_loss(history)
	x_predict = np.linspace(-4, 4, 100)
	y_predict = solver.predict(x_predict)
	x_exact = np.linspace(-4, 4, 200)
	y_exact = schrodinger(n, x_exact)
	plt.figure(1)
	plt.plot(x_exact, y_exact, color = "C1", label = "Exact solution")
	plt.plot(x_predict, y_predict, ".", color = "C0", markersize = 3, label = "Neural network solution")
	plt.xlabel("$x$", fontsize = 15)
	plt.ylabel("$\psi(x)$", fontsize = 15)
	plt.xlim(min(x_predict), max(x_predict))
	plt.legend()
	plt.show()
	

	#----------------------------------
	#-----------Figure-3---------------
	#-----------n = 1------------------
	#----------------------------------
	
	order = 2
	diffeq = "schrodinger"
	n = 1
	x = np.linspace(-5, 5, 100)
	x0, y0 = 2, schrodinger(n, 2)
	dx0, dy0 = -2, schrodinger(n, -2)
	initial_condition = ((x0, y0), (dx0, dy0))
	epochs = 50000
	architecture = [50]
	initializer = Dict["initializer"]["GlorotNormal"]
	activation = Dict["activation"]["sigmoid"]
	optimizer = Dict["optimizer"]["Adam"]
	prediction_save = False
	weights_save = False


	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	#Training
	history = solver.train()
	epoch, loss = solver.get_loss(history)


	#Plot the exact and the neural net solution
	x_predict = np.linspace(-4, 4, 100)
	y_predict = solver.predict(x_predict)
	y_exact = schrodinger(n, x_predict)
	print(solver.mean_relative_error(y_predict, y_exact))
	solver.plot_solution(x_predict, y_predict, y_exact)
	

#--------------------------------------------------------------------
#-----------------BURST----------------------------------------------
#--------------------------------------------------------------------

if __name__ == "__main__":

	#Exact solution
	def burst(n, x):
		n = abs(n)
		return np.sqrt(1 + x**2)/n * np.cos(n * np.arctan(x))

	#----------------------------------
	#----------Figure-4----------------
	#----------n = 10------------------
	#----------------------------------
	
	order = 2
	diffeq = "burst"
	n = 10
	x = np.linspace(-7, 7, 300)
	x0, y0 = 1.5, burst(n, 1.5)
	dx0, dy0 = 3, burst(n, 3)
	initial_condition = ((x0, y0), (dx0, dy0))
	epochs = 200000
	architecture = [30, 30, 30]
	initializer = Dict["initializer"]["GlorotNormal"]
	activation = Dict["activation"]["tanh"]
	optimizer = Dict["optimizer"]["Adamax"]
	prediction_save = False
	weights_save = False


	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	#Training
	history = solver.train()
	epoch, loss = solver.get_loss(history)


	#Plot the exact and the neural net solution
	x_predict = x
	y_predict = solver.predict(x_predict)
	y_exact = burst(n, x_predict)
	print(solver.mean_relative_error(y_predict, y_exact))
	solver.plot_solution(x_predict, y_predict, y_exact)
	


#--------------------------------------------------------------------
#-----------------Figure-5-------------------------------------------
#--------------------------------------------------------------------

if __name__ == "__main__":
	
	order = 2
	diffeq = "schrodinger"
	n = 5
	x0, y0 = 4, schrodinger(n, 4)
	dx0, dy0 = -4, schrodinger(n, -4)
	initial_condition = ((x0, y0), (dx0, dy0))
	epochs = 400000
	architecture = [20, 20]
	initializer = Dict["initializer"]["GlorotNormal"]
	activation = Dict["activation"]["tanh"]
	optimizer = Dict["optimizer"]["Adamax"]
	prediction_save = False
	weights_save = False

	#Evenly spaced sample
	x = np.linspace(-5, 5, 100)
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch, loss = solver.get_loss(history)
	x_predict_ESS = x
	y_predict_ESS = solver.predict(x_predict_ESS)
	y_exact_ESS = schrodinger(n, x_predict_ESS)
	error_ESS = solver.relative_error(y_predict_ESS, y_exact_ESS)

	#Random uniform sample
	x = np.random.uniform(low = -5, high = 5, size = 100)
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch, loss = solver.get_loss(history)
	x_predict_RUS = x
	y_predict_RUS = solver.predict(x_predict_RUS)
	y_exact_RUS = schrodinger(n, x_predict_RUS)
	error_RUS = solver.relative_error(y_predict_RUS, y_exact_RUS)

	#Random gaussian sample
	x = np.random.normal(loc = 0, scale = 1, size = 100)
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch, loss = solver.get_loss(history)
	x_predict_RGS = x
	y_predict_RGS = solver.predict(x_predict_RGS)
	y_exact_RGS = schrodinger(n, x_predict_RGS)
	error_RGS = solver.relative_error(y_predict_RGS, y_exact_RGS)

	#Plot
	x_exact = np.linspace(-5, 5, 500)
	y_exact = schrodinger(n, x_exact)
	fig = plt.figure()

	axe1 = fig.add_axes([0.17, 0.35, 0.75, 0.6])
	axe1.set_ylabel("$\psi(x)$", fontsize = 15)
	axe1.set_xticks([])
	axe1.set_xlim(min(x_exact), max(x_exact))
	axe2 = fig.add_axes([0.17, 0.1, 0.75, 0.25])
	axe2.set_xlim(min(x_exact), max(x_exact))
	axe2.set_ylabel("Relative \n error, $\\frac{|\\Delta \psi|}{|\psi|}$", fontsize = 15)
	axe2.set_xlabel("$x$", fontsize = 15)
	axe2.set_yscale('log')

	axe1.plot(x_exact, y_exact, color = "C1", label = "Exact solution")
	axe1.plot(x_predict_ESS, y_predict_ESS, ".", color = "C0", label = "Evenly spaced", markersize = 3)
	axe1.plot(x_predict_RUS, y_predict_RUS, ".", color = "C2", label = "Random uniform", markersize = 3)
	axe1.plot(x_predict_RGS, y_predict_RGS, ".", color = "C3", label = "Random gaussian", markersize = 3)
	axe1.legend()

	axe2.plot(x_predict_ESS, error_ESS, ".", color = "C0", markersize = 5)
	axe2.plot(x_predict_RUS, error_RUS, ".", color = "C2", markersize = 5)
	axe2.plot(x_predict_RGS, error_RGS, ".", color = "C3", markersize = 5)
	plt.show()
	

#--------------------------------------------------------------------
#-----------------Figure-6------------------------------------------
#--------------------------------------------------------------------

if __name__ == "__main__":

	order = 2
	diffeq = "schrodinger"
	n = 5
	x = np.linspace(-4, 4, 100)
	x0, y0 = 3, schrodinger(n, 3)
	dx0, dy0 = -3, schrodinger(n, -3)
	initial_condition = ((x0, y0), (dx0, dy0))
	epochs = 500000
	architecture = [30, 30, 30]
	initializer = Dict["initializer"]["GlorotNormal"]
	activation = Dict["activation"]["tanh"]
	optimizer = Dict["optimizer"]["Adamax"]
	prediction_save = False
	weights_save = False


	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch, loss = solver.get_loss(history)

	x_predict = x
	y_predict = solver.predict(x_predict)
	error_predict = solver.relative_error(y_predict, schrodinger(n, x_predict))

	x_1 = np.linspace(0, 2, 40)
	y_1 = solver.predict(x_1)
	error_1 = abs( np.reshape(y_1, len(x_1)) - np.reshape(schrodinger(n, x_1), len(x_1)) )/abs(np.reshape(schrodinger(n, x_1), len(x_1)))

	x_2 = np.linspace(-6, -4, 30)
	y_2 = solver.predict(x_2)
	error_2 = abs( np.reshape(y_2, len(x_2)) - np.reshape(schrodinger(n, x_2), len(x_2)) )/abs(np.reshape(schrodinger(n, x_2), len(x_2)))

	x_3 = np.linspace(4, 6, 30)
	y_3 = solver.predict(x_3)
	error_3 = abs( np.reshape(y_3, len(x_3)) - np.reshape(schrodinger(n, x_3), len(x_3)) )/abs(np.reshape(schrodinger(n, x_3), len(x_3)))

	#Plot
	x_exact = np.linspace(-6, 6, 500)
	y_exact = schrodinger(n, x_exact)
	fig = plt.figure()

	axe1 = fig.add_axes([0.17, 0.35, 0.75, 0.6])
	axe1.set_ylabel("$\psi(x)$", fontsize = 15)
	axe1.set_xticks([])
	axe1.set_xlim(min(x_exact), max(x_exact))
	axe2 = fig.add_axes([0.17, 0.1, 0.75, 0.25])
	axe2.set_xlim(min(x_exact), max(x_exact))
	axe2.set_ylabel("Relative \n error, $\\frac{|\\Delta \psi|}{|\psi|}$", fontsize = 15)
	axe2.set_xlabel("$x$", fontsize = 15)
	axe2.set_yscale('log')

	axe1.plot(x_exact, y_exact, color = "C1", label = "Exact solution")
	axe1.plot(x_predict, y_predict, ".", color = "C0", label = "Training sample", markersize = 3)
	axe1.plot(x_1, y_1, ".", color = "C2", label = "Prediction inside \n the training domain", markersize = 3)
	axe1.plot(x_2, y_2, ".", color = "C3", label = "Prediction outside \n the training domain", markersize = 3)
	axe1.plot(x_3, y_3, ".", color = "C3", markersize = 3)
	axe1.legend()

	axe2.plot(x_predict, error_predict, ".", color = "C0", markersize = 4)
	axe2.plot(x_1, error_1, ".", color = "C2", markersize = 4)
	axe2.plot(x_2, error_2, ".", color = "C3", markersize = 4)
	axe2.plot(x_3, error_3, ".", color = "C3", markersize = 4)
	plt.show()


#--------------------------------------------------------------------
#-----------------Figure-7-------------------------------------------
#--------------------------------------------------------------------

if __name__ == "__main__":
	
	#----------------------------------
	#----------First-order-ode---------
	#----------------------------------
	
	order = 1
	diffeq = "first order ode"
	x = np.linspace(0, 2, 100)
	initial_condition = (0, 0)
	epochs = 40000
	initializer = Dict["initializer"]["GlorotNormal"]
	activation = Dict["activation"]["sigmoid"]
	optimizer = Dict["optimizer"]["Adam"]
	prediction_save = False
	weights_save = False

	#Average the loss every avr value to have a smooth function
	#epochs should be a multiple of avr epochs/avr = int
	avr = 5

	#Neurons
	architecture = [10]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch_10, loss_10 = solver.get_loss(history)
	epoch_10 = np.asarray(epoch_10)
	epoch_10 = np.mean(epoch_10.reshape(-1, avr), axis = 1)
	loss_10 = np.asarray(loss_10)
	loss_10 = np.mean(loss_10.reshape(-1, avr), axis = 1)
	architecture = [20]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch_20, loss_20 = solver.get_loss(history)
	epoch_20 = np.asarray(epoch_20)
	epoch_20 = np.mean(epoch_20.reshape(-1, avr), axis = 1)
	loss_20 = np.asarray(loss_20)
	loss_20 = np.mean(loss_20.reshape(-1, avr), axis = 1)
	architecture = [50]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch_50, loss_50 = solver.get_loss(history)
	epoch_50 = np.asarray(epoch_50)
	epoch_50 = np.mean(epoch_50.reshape(-1, avr), axis = 1)
	loss_50 = np.asarray(loss_50)
	loss_50 = np.mean(loss_50.reshape(-1, avr), axis = 1)
	architecture = [100]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch_100, loss_100 = solver.get_loss(history)
	epoch_100 = np.asarray(epoch_100)
	epoch_100 = np.mean(epoch_100.reshape(-1, avr), axis = 1)
	loss_100 = np.asarray(loss_100)
	loss_100 = np.mean(loss_100.reshape(-1, avr), axis = 1)
	architecture = [1000]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch_1000, loss_1000 = solver.get_loss(history)
	epoch_1000 = np.asarray(epoch_1000)
	epoch_1000 = np.mean(epoch_1000.reshape(-1, avr), axis = 1)
	loss_1000 = np.asarray(loss_1000)
	loss_1000 = np.mean(loss_1000.reshape(-1, avr), axis = 1)

	#Hidden layers
	architecture = [20]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch__20, loss__20 = solver.get_loss(history)
	epoch__20 = np.asarray(epoch__20)
	epoch__20 = np.mean(epoch__20.reshape(-1, avr), axis = 1)
	loss__20 = np.asarray(loss__20)
	loss__20 = np.mean(loss__20.reshape(-1, avr), axis = 1)
	architecture = [20, 20]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch__20_20, loss__20_20 = solver.get_loss(history)
	epoch__20_20 = np.asarray(epoch__20_20)
	epoch__20_20 = np.mean(epoch__20_20.reshape(-1, avr), axis = 1)
	loss__20_20 = np.asarray(loss__20_20)
	loss__20_20 = np.mean(loss__20_20.reshape(-1, avr), axis = 1)
	architecture = [20, 20, 20]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch__20_20_20, loss__20_20_20 = solver.get_loss(history)
	epoch__20_20_20 = np.asarray(epoch__20_20_20)
	epoch__20_20_20 = np.mean(epoch__20_20_20.reshape(-1, avr), axis = 1)
	loss__20_20_20 = np.asarray(loss__20_20_20)
	loss__20_20_20 = np.mean(loss__20_20_20.reshape(-1, avr), axis = 1)
	architecture = [20, 20, 20, 20]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch__20_20_20_20, loss__20_20_20_20 = solver.get_loss(history)
	epoch__20_20_20_20 = np.asarray(epoch__20_20_20_20)
	epoch__20_20_20_20 = np.mean(epoch__20_20_20_20.reshape(-1, avr), axis = 1)
	loss__20_20_20_20 = np.asarray(loss__20_20_20_20)
	loss__20_20_20_20 = np.mean(loss__20_20_20_20.reshape(-1, avr), axis = 1)
	
	#Plot
	fig = plt.figure(figsize = (9.5, 4.6))
	ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[])
	ax1.set_xlim(epoch_10[500], epoch_10[-1])
	ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
	ax2.set_xlim(epoch_10[500], epoch_10[-1])
	ax2.set_xticks([10000, 20000, 30000, 40000])
	ax2.set_xticklabels(["$10^4$", "$2.10^4$", "$3.10^4$", "$4.10^4$"])
	ax2.set_xlabel("Number of epochs", fontsize = 15)
	fig.text(0.02, 0.5, "Loss function, $\mathcal{L}$", fontsize = 15, va = 'center', rotation='vertical')

	ax1.semilogy(epoch_1000, loss_1000, label = "(1000)", color = "C4")
	ax1.semilogy(epoch_100, loss_100, label = "(100)", color = "C3")
	ax1.semilogy(epoch_50, loss_50, label = "(50)", color = "C2")
	ax1.semilogy(epoch_20, loss_20, label = "(20)", color = "C1")
	ax1.semilogy(epoch_10, loss_10, label = "(10)", color = "C0")
	ax1.legend()

	ax2.semilogy(epoch__20_20_20_20, loss__20_20_20_20, label = "(20, 20, 20, 20)", color = "C3")
	ax2.semilogy(epoch__20_20_20, loss__20_20_20, label = "(20, 20, 20)", color = "C2")
	ax2.semilogy(epoch__20_20, loss__20_20, label = "(20, 20)", color = "C1")
	ax2.semilogy(epoch__20, loss__20, label = "(20)", color = "C0")
	ax2.legend()

	plt.show()
	

	#----------------------------------
	#----------Schrodinger-------------
	#----------------------------------
	
	order = 2
	n = 2
	diffeq = "schrodinger"
	x = np.linspace(-5, 5, 100)
	x0, y0 = 2, schrodinger(n, 2)
	dx0, dy0 = -2, schrodinger(n, -2)
	initial_condition = ((x0, y0), (dx0, dy0))
	epochs = 100000
	initializer = Dict["initializer"]["GlorotNormal"]
	activation = Dict["activation"]["sigmoid"]
	optimizer = Dict["optimizer"]["Adam"]
	prediction_save = False
	weights_save = False

	#Average the loss every avr value to have a smooth function
	#epochs should be a multiple of avr epochs/avr = int
	avr = 5

	#Neurons
	architecture = [10]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch_10, loss_10 = solver.get_loss(history)
	epoch_10 = np.asarray(epoch_10[5000:])
	epoch_10 = np.mean(epoch_10.reshape(-1, avr), axis = 1)
	loss_10 = np.asarray(loss_10[5000:])
	loss_10 = np.mean(loss_10.reshape(-1, avr), axis = 1)
	architecture = [20]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch_20, loss_20 = solver.get_loss(history)
	epoch_20 = np.asarray(epoch_20[5000:])
	epoch_20 = np.mean(epoch_20.reshape(-1, avr), axis = 1)
	loss_20 = np.asarray(loss_20[5000:])
	loss_20 = np.mean(loss_20.reshape(-1, avr), axis = 1)
	architecture = [50]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch_50, loss_50 = solver.get_loss(history)
	epoch_50 = np.asarray(epoch_50[5000:])
	epoch_50 = np.mean(epoch_50.reshape(-1, avr), axis = 1)
	loss_50 = np.asarray(loss_50[5000:])
	loss_50 = np.mean(loss_50.reshape(-1, avr), axis = 1)
	architecture = [100]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch_100, loss_100 = solver.get_loss(history)
	epoch_100 = np.asarray(epoch_100[5000:])
	epoch_100 = np.mean(epoch_100.reshape(-1, avr), axis = 1)
	loss_100 = np.asarray(loss_100[5000:])
	loss_100 = np.mean(loss_100.reshape(-1, avr), axis = 1)
	architecture = [1000]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch_1000, loss_1000 = solver.get_loss(history)
	epoch_1000 = np.asarray(epoch_1000[5000:])
	epoch_1000 = np.mean(epoch_1000.reshape(-1, avr), axis = 1)
	loss_1000 = np.asarray(loss_1000[5000:])
	loss_1000 = np.mean(loss_1000.reshape(-1, avr), axis = 1)

	#Hidden layers
	architecture = [20, 20]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch__20_20, loss__20_20 = solver.get_loss(history)
	epoch__20_20 = np.asarray(epoch__20_20[5000:])
	epoch__20_20 = np.mean(epoch__20_20.reshape(-1, avr), axis = 1)
	loss__20_20 = np.asarray(loss__20_20[5000:])
	loss__20_20 = np.mean(loss__20_20.reshape(-1, avr), axis = 1)
	architecture = [20, 20, 20]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch__20_20_20, loss__20_20_20 = solver.get_loss(history)
	epoch__20_20_20 = np.asarray(epoch__20_20_20[5000:])
	epoch__20_20_20 = np.mean(epoch__20_20_20.reshape(-1, avr), axis = 1)
	loss__20_20_20 = np.asarray(loss__20_20_20[5000:])
	loss__20_20_20 = np.mean(loss__20_20_20.reshape(-1, avr), axis = 1)
	architecture = [20, 20, 20, 20]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch__20_20_20_20, loss__20_20_20_20 = solver.get_loss(history)
	epoch__20_20_20_20 = np.asarray(epoch__20_20_20_20[5000:])
	epoch__20_20_20_20 = np.mean(epoch__20_20_20_20.reshape(-1, avr), axis = 1)
	loss__20_20_20_20 = np.asarray(loss__20_20_20_20[5000:])
	loss__20_20_20_20 = np.mean(loss__20_20_20_20.reshape(-1, avr), axis = 1)
	
	#Plot
	fig = plt.figure(figsize = (9.5, 4.6))
	ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[])
	ax1.set_xlim(epoch_10[2000], epoch_10[-1])
	ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
	ax2.set_xlim(epoch_10[2000], epoch_10[-1])
	ax2.set_xticks([20000, 40000, 60000, 80000, 100000])
	ax2.set_xticklabels(["$2.10^4$", "$4.10^4$", "$6.10^4$", "$8.10^4$", "$10^5$"])
	ax2.set_xlabel("Number of epochs", fontsize = 15)
	fig.text(0.02, 0.5, "Loss function, $\mathcal{L}$", fontsize = 15, va = 'center', rotation='vertical')

	ax1.semilogy(epoch_1000[2000:], loss_1000[2000:], label = "(1000)", color = "C4")
	ax1.semilogy(epoch_100[2000:], loss_100[2000:], label = "(100)", color = "C3")
	ax1.semilogy(epoch_50[2000:], loss_50[2000:], label = "(50)", color = "C2")
	ax1.semilogy(epoch_20[2000:], loss_20[2000:], label = "(20)", color = "C1")
	ax1.semilogy(epoch_10[2000:], loss_10[2000:], label = "(10)", color = "C0")
	ax1.legend()

	ax2.semilogy(epoch__20_20_20_20[2000:], loss__20_20_20_20[2000:], label = "(20, 20, 20, 20)", color = "C3")
	ax2.semilogy(epoch__20_20_20[2000:], loss__20_20_20[2000:], label = "(20, 20, 20)", color = "C2")
	ax2.semilogy(epoch__20_20[2000:], loss__20_20[2000:], label = "(20, 20)", color = "C1")
	ax2.semilogy(epoch_20[2000:], loss_20[2000:], label = "(20)", color = "C0")
	ax2.legend()

	plt.show()
	

	#----------------------------------
	#----------Burst-------------------
	#----------------------------------
	
	order = 2
	n = 10
	diffeq = "burst"
	x = np.linspace(-7, 7, 300)
	x0, y0 = 1.5, burst(n, 1.5)
	dx0, dy0 = 3, burst(n, 3)
	initial_condition = ((x0, y0), (dx0, dy0))
	epochs = 200000
	initializer = Dict["initializer"]["GlorotNormal"]
	activation = Dict["activation"]["tanh"]
	optimizer = Dict["optimizer"]["Adamax"]
	prediction_save = False
	weights_save = False

	#Average the loss every avr value to have a smooth function
	#epochs should be a multiple of avr epochs/avr = int
	avr = 5

	#Neurons
	architecture = [10]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch_10, loss_10 = solver.get_loss(history)
	epoch_10 = np.asarray(epoch_10[5000:])
	epoch_10 = np.mean(epoch_10.reshape(-1, avr), axis = 1)
	loss_10 = np.asarray(loss_10[5000:])
	loss_10 = np.mean(loss_10.reshape(-1, avr), axis = 1)
	architecture = [20]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch_20, loss_20 = solver.get_loss(history)
	epoch_20 = np.asarray(epoch_20[5000:])
	epoch_20 = np.mean(epoch_20.reshape(-1, avr), axis = 1)
	loss_20 = np.asarray(loss_20[5000:])
	loss_20 = np.mean(loss_20.reshape(-1, avr), axis = 1)
	architecture = [50]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch_50, loss_50 = solver.get_loss(history)
	epoch_50 = np.asarray(epoch_50[5000:])
	epoch_50 = np.mean(epoch_50.reshape(-1, avr), axis = 1)
	loss_50 = np.asarray(loss_50[5000:])
	loss_50 = np.mean(loss_50.reshape(-1, avr), axis = 1)
	architecture = [100]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch_100, loss_100 = solver.get_loss(history)
	epoch_100 = np.asarray(epoch_100[5000:])
	epoch_100 = np.mean(epoch_100.reshape(-1, avr), axis = 1)
	loss_100 = np.asarray(loss_100[5000:])
	loss_100 = np.mean(loss_100.reshape(-1, avr), axis = 1)
	architecture = [1000]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch_1000, loss_1000 = solver.get_loss(history)
	epoch_1000 = np.asarray(epoch_1000[5000:])
	epoch_1000 = np.mean(epoch_1000.reshape(-1, avr), axis = 1)
	loss_1000 = np.asarray(loss_1000[5000:])
	loss_1000 = np.mean(loss_1000.reshape(-1, avr), axis = 1)

	#Hidden layers
	architecture = [20, 20]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch__20_20, loss__20_20 = solver.get_loss(history)
	epoch__20_20 = np.asarray(epoch__20_20[5000:])
	epoch__20_20 = np.mean(epoch__20_20.reshape(-1, avr), axis = 1)
	loss__20_20 = np.asarray(loss__20_20[5000:])
	loss__20_20 = np.mean(loss__20_20.reshape(-1, avr), axis = 1)
	architecture = [20, 20, 20]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch__20_20_20, loss__20_20_20 = solver.get_loss(history)
	epoch__20_20_20 = np.asarray(epoch__20_20_20[5000:])
	epoch__20_20_20 = np.mean(epoch__20_20_20.reshape(-1, avr), axis = 1)
	loss__20_20_20 = np.asarray(loss__20_20_20[5000:])
	loss__20_20_20 = np.mean(loss__20_20_20.reshape(-1, avr), axis = 1)
	architecture = [20, 20, 20, 20]
	solver = ODEsolver(order, diffeq, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save, weights_save)
	history = solver.train()
	epoch__20_20_20_20, loss__20_20_20_20 = solver.get_loss(history)
	epoch__20_20_20_20 = np.asarray(epoch__20_20_20_20[5000:])
	epoch__20_20_20_20 = np.mean(epoch__20_20_20_20.reshape(-1, avr), axis = 1)
	loss__20_20_20_20 = np.asarray(loss__20_20_20_20[5000:])
	loss__20_20_20_20 = np.mean(loss__20_20_20_20.reshape(-1, avr), axis = 1)
	
	#Plot
	fig = plt.figure(figsize = (9.5, 4.6))
	ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[])
	ax1.set_xlim(epoch_10[2000], epoch_10[-1])
	ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
	ax2.set_xlim(epoch_10[2000], epoch_10[-1])
	ax2.set_xticks([50000, 100000, 150000, 200000])
	ax2.set_xticklabels(["$5.10^4$", "$10^5$", "$15.10^4$", "$2.10^5$"])
	ax2.set_xlabel("Number of epochs", fontsize = 15)
	fig.text(0.02, 0.5, "Loss function, $\mathcal{L}$", fontsize = 15, va = 'center', rotation='vertical')

	ax1.semilogy(epoch_1000[2000:], loss_1000[2000:], label = "(1000)", color = "C4")
	ax1.semilogy(epoch_100[2000:], loss_100[2000:], label = "(100)", color = "C3")
	ax1.semilogy(epoch_50[2000:], loss_50[2000:], label = "(50)", color = "C2")
	ax1.semilogy(epoch_20[2000:], loss_20[2000:], label = "(20)", color = "C1")
	ax1.semilogy(epoch_10[2000:], loss_10[2000:], label = "(10)", color = "C0")
	ax1.legend()

	ax2.semilogy(epoch__20_20_20_20[2000:], loss__20_20_20_20[2000:], label = "(20, 20, 20, 20)", color = "C3")
	ax2.semilogy(epoch__20_20_20[2000:], loss__20_20_20[2000:], label = "(20, 20, 20)", color = "C2")
	ax2.semilogy(epoch__20_20[2000:], loss__20_20[2000:], label = "(20, 20)", color = "C1")
	ax2.semilogy(epoch_20[2000:], loss_20[2000:], label = "(20)", color = "C0")
	ax2.legend()

	plt.show()
	




