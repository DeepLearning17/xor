import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 2])
y_hat = tf.placeholder(tf.float32, [None, 1]) # correct answers

# a fine transformation
W = tf.Variable(tf.zeros([2,2]))
c = tf.Variable(tf.zeros([2]))

hidden_layer = tf.nn.relu(tf.matmul(x, W) + c)

# output layer

w = tf.Variable(tf.zeros([1,2]))
b = tf.Variable(tf.zeros([1]))

output = tf.matmul(hidden_layer, w, transpose_b=True) + b

loss = tf.reduce_mean(tf.square(y_hat - output), reduction_indices=[1])

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss, var_list=[W, c, w, b])

data = {x : [[0, 0], [0,1], [1,0], [1, 1]], y_hat: [[a] for a in [0, 1, 1, 0]]}

with tf.Session() as session:
    tf.global_variables_initializer().run(session=session)
    for i in range(1000):
        res = session.run(train_step, feed_dict=data)

    result = session.run(output, data)
    print("W", session.run(W))
    print("c", session.run(c))

    print("w", session.run(w))
    print("b", session.run(b))

    print("--")
    print(result)
