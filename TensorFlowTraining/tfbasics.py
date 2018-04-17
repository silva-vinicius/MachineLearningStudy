import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

#defining and modelling a computation graph.
#efficient multiplication
result = tf.multiply(x1,x2)

#this will print the defined computation model, not its output
print (result)

#we dont have to worry about closing the session because the following block does that for us
with tf.Session() as sess:
	print(sess.run(result))
