'''
Planning:
	
	Feed Forwarding

	input > weight > hidden l1 (activation function) > weights > 
	hidden l2 (activation function) > weights > output layer
	
	Compare output to intended output > cost function (cross entropy)
	optimization function (optimizer) > minimize cost (AdamOptimizer, SGD, AdaGrad)

	backpropagation

	feed forward + backprop = epoch
'''
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

#understanding oneHot encoding
# 10 classses, 0-9 

'''
	0=[1,0,0,0,0,0,0,0,0,0]
	1=[0,1,0,0,0,0,0,0,0,0]
	2=[0,0,1,0,0,0,0,0,0,0]
	.
	.
	.
'''

#defining the number of neurons in each hidden layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

#defining the number of possible classes in which we can attribute our classifications
n_classes = 10

#defines how many images will be dealt at once
batch_size = 100


#restricting the input format: a flattened 28x28 image
x=tf.placeholder('float', [None, 28*28])

#restricting the output format: a float value
y=tf.placeholder('float')

#builds the computation graph
def neural_network_model(data):

					#one column per neuron. Each neuron receives 784 sinapses # Matrix Z
	hidden_1_layer={'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),

					#each neuron also gets a bias, which is a constant that we will add to the summation
					'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer={'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer={'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer={'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
				  'biases': tf.Variable(tf.random_normal([n_classes]))}


	l1 = tf.add( tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	#relu = rectified linear function -> This is our activation function
	l1 = tf.nn.relu(l1)
	
	l2 = tf.add( tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add( tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output


def train_neural_network(X):

	prediction = neural_network_model(X)


	#defining our cost function
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

	#we will use the AdamOptimizer to minimize the cost function
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	#cycles feed forward + backprop
	n_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		#----------------------------training the network----------------------------
		for epoch in range(n_epochs):
			epoch_loss = 0

			#training on each batch
			for _ in range(int(mnist.train.num_examples/batch_size)):

					   #splits the data in chunks of size <batch_size>
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)

				   # c -> cost
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c

			print('Epoch ', epoch, ' completed out of ', n_epochs, ' loss: ', epoch_loss)

		#-----------------------------------------------------------------------------

		#tf.argmax -> finds the index of the maximum value
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))


		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)



