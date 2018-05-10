#tutorial available at: https://pythonprogramming.net/train-test-tensorflow-deep-learning-tutorial/
import tensorflow as tf
import numpy as np

from create_sentiment_featuresets import create_feature_sets_and_labels

train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')


#defining the number of neurons in each hidden layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

#defining the number of possible classes in which we can attribute our classifications
n_classes = 2

#defines how many images will be dealt at once
batch_size = 100


#restricting the input format: a flattened 28x28 image
x=tf.placeholder('float', [None, len(train_x[0])])

#restricting the output format: a float value
y=tf.placeholder('float')

#builds the computation graph
def neural_network_model(data):

					#one column per neuron. Each neuron receives len(train_x[0]) sinapses - Matrix Z1 - 784 rows - 500 columns
	hidden_1_layer={'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),

					#each neuron also gets a bias, which is a constant that we will add to the summation
					'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer={'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer={'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer={'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
				  'biases': tf.Variable(tf.random_normal([n_classes]))}


	#tf.add -> Matrix sum. We are adding the result of data*Z1 and layer 1 biases
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

	#cycles (feed forward + backprop)
	n_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		#----------------------------training the network----------------------------
		for epoch in range(n_epochs):
			epoch_loss = 0

			
			i=0

			while i<len(train_x):

				start = i
				end = i+batch_size

				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])




				   # c -> cost
				_, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
				epoch_loss += c
				i+=batch_size

			print('Epoch ', epoch, ' completed out of ', n_epochs, ' loss: ', epoch_loss)

		#-----------------------------------------------------------------------------

		#tf.argmax -> finds the index of the maximum value
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))


		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy: ', accuracy.eval({x:test_x, y:test_y}))


train_neural_network(x)


