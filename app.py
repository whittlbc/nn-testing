import numpy as np

# We need to build a neural network trained with backpropagation that can
# predict the following output column given these three input columns:

# Inputs    | Output
# ------------------
# [0, 0, 1] | [0]
# [0, 1, 1] | [0]
# [1, 0, 1] | [1]
# [1, 1, 1] | [1}

# Input dataset matrix (4x3)
# Rows are "training examples".
# Columns are "input nodes".
# 3 input nodes to the network and 4 training examples.
X = np.array([
	[0, 0, 1],
	[0, 1, 1],
	[1, 0, 1],
	[1, 1, 1]
])

# Output dataset matrix (4x1)
# 4 training examples with 1 output node
y = np.array([
	[0, 0, 1, 1]
]).T

# It's good practice to seed your random numbers. Your numbers will still be randomly
# distributed, but they'll be randomly distributed in exactly the same way each time you train.
# This makes it easier to see how your changes affect the network.
np.random.seed(1)

# Initialize weights randomly with a mean of 0.
# Since our network has 3 inputs and 1 output, we need a 3x1 matrix for Synapse 0.
syn0 = 2 * np.random.random((3, 1)) - 1

# Define your sigmoid function. The sigmoid function is such a good candidate for
# neural networks because it's naturally non-linear, meaning it can give us *probability* as an output.
# It maps any value to a value between 0 and 1, meaning we can use it to map numbers to probabilities.
# It's also desirable because its derivative is so easy/efficient to calculate and will also be between 0 and 1.
def sigmoid(x, deriv=False):
	# derivative of sigmoid function
	if deriv: return x * (1 - x)
	
	# regular sigmoid function: 1/(1 + e^-x)
	return 1 / (1 + np.exp(-x))


# Initialize layer1
l1 = None

# Forward propagation (aka. feed forward)
for i in xrange(10000):
	# layer0 is equal to the input dataset. Gonna process all 4 training examples
	# at the same time - aka. "full batch" training.
	l0 = X
	
	# This is our prediction step, where we let the network "try" to predict the output given the input.
	# We'll then study how it performs and then adjust it to do better for each iteration.
	predictions = np.dot(l0, syn0)
	l1 = sigmoid(predictions)
	
	# Find how off our predictions were (error)
	l1_error = y - l1
	
	# We then multiple the error by the slopes (deriv=True) in order to reduce the error of high confidence
	# predictions. Think about a sigmoid graph: If the slope is close to 0, then the network either had a
	# very high or very low value - this means the network was quite confident, one way or the other. However,
	# if the network guessed something with a higher slope (middle of graph), then it wasn't very confident.
	# We update these "wishy-washy" predictions most heavily, and we tend to leave the confident ones alone
	# by multiplying them by a number close to 0.
	l1_delta = l1_error * sigmoid(l1, deriv=True)
	
	# Update the network to be more precise with each iteration.
	syn0 += np.dot(l0.T, l1_delta)


print "Output After Training:\n{}".format(l1)
