import tensorflow as tf

# Tensorflow is designed around computation graphs.
# The values propagate along the graph from input to output

# Example

constant_node = tf.constant(3.0, tf.float32)

# This creates a node with a constant value of 3.0 with dtype (ala numpy) of float32.
# This node is a graph on its own.

# In order to perform computations, you need to create a session.

session = tf.Session()

# the session then runs the actions at each node it is provided, like so:

print(session.run([constant_node]))

### XOR ###

# from our example, the XOR graph looks roughly as follows:

# input -> relu(affine) -> perceptron -> ouptut

# affine

W = tf.constant([[1,1], [1,1]], shape=[2,2], dtype= tf.float32)
c = tf.constant([0,-1], shape=[2], dtype=tf.float32)

x = tf.placeholder(shape=[4,2], dtype=tf.float32)

affine = tf.matmul(x, W) + c

data =  {x: [[0,0], [0,1], [1,0], [1,1]]}

print("affine transform")

print(session.run(affine, data))


with_relu = tf.nn.relu(affine)


print("ReLU(affine transform)")

print(session.run(with_relu, data))


w = tf.constant([[1],[-2]], tf.float32)

b = tf.constant(0, tf.float32)

perceptron = tf.matmul(w, with_relu, transpose_a=True, transpose_b=True) + b

print("Complete network")

print(session.run(perceptron, data))

