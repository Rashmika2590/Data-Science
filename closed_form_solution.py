import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
n = 10

def getData():	
	mean = 0
	sigma = 1
	noise = mean + np.random.randn(n,1)*sigma
	x = np.random.rand(n,1)
	y = 2*x + 1 + noise
	return x, y

def plotData(x, y, pred):
	plt.scatter(x,y)
	plt.scatter(x,pred)
	plt.title('Data')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()
	
def calW(X, y):
	xtx_inv = np.linalg.inv(np.dot(X.T,X))
	xty = np.dot(X.T, y)
	w = np.dot(xtx_inv, xty)
	return w

def calPred(X, w):
	return np.dot(X, w)

x, y = getData() 
ones = np.ones([n,1])
X = np.append(x, ones, axis = 1) 
w = calW(X, y)
pred = calPred(X,w)
plotData(x, y, pred)









