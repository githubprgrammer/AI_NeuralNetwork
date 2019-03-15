import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt

# builds a network with specified number of units
def build_network(num_inputs, num_hidden, num_outputs):
    np.random.seed(42)
    W1 = np.random.rand(num_inputs, num_hidden)*0.01
    B1 = np.zeros((1,num_hidden))
    W2 = np.random.rand(num_hidden, num_outputs)*0.01
    B2 = np.zeros((1,num_outputs))
    network = { 'W1': W1, 'B1': B1, 'W2': W2, 'B2': B2}
    return network


# The logistic function.
# Should return a vector/matrix with the same dimensions as x.
def sigmoid(x):
    return 1/(1+ np.exp(x))


# the derivative of the activation function
# implementing an using this might lead to a more readable code
def d_sigmoid(x):
    return x*(1-x)



# calculate networks output given input X
# returns for each sample in X the corresponding network's output y_hat
def forward_propagate(network, X):
    W1 = network['W1']
    W2 = network['W2']
    B1 = network['B1']
    B2 = network['B2']
    y_hat = sigmoid(sigmoid(X@W1 + B1)@W2 + B2)
    return y_hat


# evaluate cost over a whole data set
# this will be computed during training to track progress
def cost(network, dataset):
    X = dataset['X']
    Y = dataset['Y']
    y_hat = forward_propagate(network, X)
    n = len(X)
    mse = (1/2*n) * np.mean((y_hat - Y)**2)
    return mse


def train(network, dataset, max_epochs):
    # the dataset is a list of inputs and targets
    X = dataset['X']
    Y = dataset['Y']

    W1 = network['W1']
    B1 = network['B1']
    W2 = network['W2']
    B2 = network['B2']

    for epoch in np.arange(0,max_epochs):

        # ###
        # Forward propagation
        # ###



        # ###
        # Backpropagation
        # ###



        dW2 = None # To be implemented
        db2 = None # To be implemented

        dW1 = None # To be implemented
        db1 = None # To be implemented

        # ###
        # Update the networks parameters here
        # ###

        alpha = 0.03 #learning rate
        network['W1'] = network['W1'] - None # To be implemented
        network['B1'] = network['B1'] - None # To be implemented
        network['W2'] = network['W2'] - None # To be implemented
        network['B2'] = network['B2'] - None # To be implemented


        # some printouts
        # so we have some idea what the network is doing
        if (epoch%100 == 0):
            cost_temp = cost(network, dataset)
            print('>epoch=%d, cost=%.6f' % (epoch, cost_temp))

    # returning the trained network to plot results
    return network

# returns the from the network assigned classes to the data given in X
# returned classes are either zero or one
def classify(network, X):
    pass # To be implemented

##### ##### #### ##### ##### #####
##### ##### DATA ##### ##### #####
##### ##### #### ##### ##### #####

# normalizing each column of x
def normalize(x):
    return (x - np.mean(x,0)) / np.std(x,0)

# BLOBS data
X, Y = sklearn.datasets.make_blobs(n_samples=400, centers=2, n_features=2,random_state=0)
X = normalize(X)
dataset_blobs = {'X': X, 'Y': Y}

##### ##### #### ##### ##### #####
##### ##### PLOT ##### ##### #####
##### ##### #### ##### ##### #####
def plot_summary(net, dataset):
    # area to be evaluated
    maxs = dataset['X'].max(axis=0)*1.1
    mins = dataset['X'].min(axis=0)*1.1
    x1 = np.linspace(mins[0],maxs[0], 400)
    x2 = np.linspace(mins[1], maxs[1], 400)
    x1v, x2v = np.meshgrid(x1,x2)
    # predict classes
    Z = classify(net,np.dstack(np.meshgrid(x1,x2)))
    Z = Z.reshape(len(x1),len(x2))
    # Plot the contour and training examples
    plt.contourf(x1v, x2v, Z, cmap='Paired')
    plt.scatter(dataset['X'][:, 0], dataset['X'][:, 1], c=dataset['Y'], cmap='Paired')
    plt.title("Network's Decision Landscape")
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

##### ##### ### ##### ##### #####
##### ##### RUN ##### ##### #####
##### ##### ### ##### ##### #####

# this will be executed upon calling the script

network = build_network(2, 4, 1)
trainset = dataset_blobs
print(cost(network, trainset))
network = train(network, trainset, 1000)
plot_summary(network, trainset)


