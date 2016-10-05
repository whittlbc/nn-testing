import numpy as np

# Input dataset matrix (4x3)
X = np.array([
	[0, 0, 1],
	[0, 1, 1],
	[1, 0, 1],
	[1, 1, 1]
])

# Output dataset matrix (4x1)
y = np.array([
	[0, 0, 1, 1]
]).T

# seed random number generator
np.random.seed(1)

# initialize weights randomly with a mean of 0 (hence then 2*x - 1 format)
syn0 = 2 * np.random.random((3, 1)) - 1


# Define your sigmoid function
def sigmoid(x, deriv=False):
	# derivative of sigmoid function
	if deriv:
		return x * (1 - x)
	
	# regular sigmoid function: 1/(1 + e^-x)
	return 1 / (1 + np.exp(-x))


# Initialize variable for layer1
l1 = None

# forward propagation
for i in xrange(10000):
	# layer0 is equal to the input dataset
	l0 = X
	
	# layer1 (aka. the hidden layer) is the value returned by the sigmoid
	# function when passed our input dataset * our random weights in syn0
	l1 = sigmoid(np.dot(l0, syn0))
	
	# figure out how much we differed from our output dataset (error)
	l1_error = y - l1
	
	# multiply our error by the slope (derivative) of the sigmoid function at the values in l1
	l1_delta = l1_error * sigmoid(l1, True)
	
	# update our weights
	syn0 += np.dot(l0.T, l1_delta)

print "Output After Training:\n{}".format(l1)
