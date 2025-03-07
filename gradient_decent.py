import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# # Function to generate synthetic data (same as before)
# def getData(n=10):
#     mean = 0
#     sigma = 1
#     noise = mean + np.random.randn(n,1) * sigma
#     x = np.random.rand(n, 1)
#     y = 2 * x + 1 + noise
#     return x, y

# # Function to plot the data and the prediction line
# def plotData(x, y, pred):
#     plt.scatter(x, y, label='Data Points')
#     plt.plot(x, pred, color='red', label='Prediction Line')
#     plt.title('Linear Regression with Gradient Descent')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend()
#     plt.show()

# # Gradient Descent algorithm for Linear Regression
# def gradientDescent(X, y, w, learning_rate, iterations):
#     m = len(y)  # Number of data points
#     history = []  # To track the loss over iterations
    
#     for i in range(iterations):
#         # Compute the predictions
#         predictions = np.dot(X, w)
        
#         # Compute the cost (Mean Squared Error)
#         cost = (1/(2*m)) * np.sum((predictions - y) ** 2)
#         history.append(cost)
        
#         # Compute the gradients
#         gradients = (1/m) * np.dot(X.T, predictions - y)
        
#         # Update weights using the gradient and learning rate
#         w = w - learning_rate * gradients
        
#     return w, history

# # Main program execution
# x, y = getData(100)  # Generate data with 100 points

# # Add a column of ones to x for the bias term (intercept)
# ones = np.ones([x.shape[0], 1])
# X = np.hstack([x, ones])

# # Initial weights (w0, w1)
# w_initial = np.zeros([X.shape[1], 1])

# # Learning rate and number of iterations
# learning_rate = 0.1
# iterations = 1000

# # Apply gradient descent
# w_optimal, history = gradientDescent(X, y, w_initial, learning_rate, iterations)

# # Calculate predictions using the optimal weights
# predictions = np.dot(X, w_optimal)

# # Plot the original data and the fitted line
# plotData(x, y, predictions)

# # Optionally: plot the cost history to see how the model converges
# plt.plot(range(iterations), history, label='Cost')
# plt.title('Cost Function Convergence')
# plt.xlabel('Iterations')
# plt.ylabel('Cost')
# plt.show()

# print("Optimal weights (w):", w_optimal)
#////////////////////////////////////////////////////////////////////////////////////////////////////////////

# a= np.array([1,2,3,4,5,6])
# print(a.shape)


# b= np.reshape(a, [2,3])
# print(b)

# c= np.reshape(b, [1,-1])#indexes 6 = -1
# print(c)

# print('///////////////////////////////////////////')
# print()


df =pd.read_csv('data.csv') 

def plotData(x,y):
    plt.scatter(x,y)
    plt.title('Data')
    plt.xlabel('X (input): Scores')
    plt.ylabel('Y (input): Houres')
    plt.show()
    
def plotLoss(L):
    plt.plot(np.arange(len(L)), L, 'r-')
    plt.show()

#read data
X_ori = np.array(df['Hours']).reshape([-1,1])
t = np.array(df['Scores']).reshape([-1,1])
#plotData(X_ori,t)

X=np.append(X_ori,np.ones_like(X_ori),axis=1)
N,D=X.shape
w=np.zeros((D,1))

lr=0.001
bs = 2
L_arr=[]
for i in range(100):
    y=np.dot(X, w)          #calculate the predictions
    
    ind = np.random.choice(N, bs, replace=False)
    X_s = X[ind, :]
    y_s = y[ind]
    t_s = t[ind]
    print(X.shape, X_s.shape)
    dw=np.dot(X_s.T, y_s-t_s)/N  
    
    
    
    #dw=np.dot(X.T, y-t)/N   #claculate the gradient
    w=2-lr*dw               #update the weights
    L=(np.mean(y-t)**2)/2     #calculate the loss value
    L_arr.append(L)
    
plotLoss(L_arr)

y=np.dot(X, w)
re = np.append(y,t,axis=1)
print(re)