from keras.models import Sequential
import numpy as np

class deepQCartSolver():

	def __init__(self, nEpisodes=1000, minScoreToWin=200, gamma=1.0, epsilon=1.0, minEpsilon = 0.01, epsilonDecay=0.99, alpha=0.01, alphaDecay = 0.01, batchSize=50):

		self.env = gym.make('CartPole-v0')
		self.nEpisodes = nEpisodes
		self.minScoreToWin = minScoreToWin
		self.memory = deque(maxlen=10000) #This dataset will feed our neural network with data
		self.gamma=gamma
		self.epsilon=epsilon 
		self.minEpsilon = minEpsilon
		self.epsilonDecay=epsilonDecay 
		self.alpha=alpha 
		self.alphaDecay = alphaDecay
		self.batchSize=batchSize

		self.nnModel = self.initNNModel()


	#layerNeurons -> input_nodes, hiddem_nodes, hidden_nodes, output_node
	#activationFunc -> hidden_nodes activation, hidden_nodes activation, output_activation
	def initNNModel(self, layerNeurons=[4, 24, 24, 2], activationFunc = ['relu', 'relu', 'linear'], lossFunc='mse'):

		neuralNet = Sequential()

		for i in range(len(layerNeurons)-1):

			if i==0:

				#adds the input layer and the first hidden layer
				model.add(Dense(layerNeurons[i+1], input_dim=layerNeurons[i], activation=activationFunc[i]))
			
			else:
				if i!=(len(layerNeurons) - 2):

					#adds all the other hidden layers
					model.add(Dense(layerNeurons[i+1], activation=activationFunc[i+1]))

				else:
					#adds the last layer
					model.add(Dense(layerNeurons[i+1], activation=activationFunc[i+1]))

		model.compile(loss=lossFunc, optimizer=Adam(lr=self.alpha))

		return model


	#transforms a line vector into a column vector so that keras understands it.
	def reshapeState(self, state):
		return np.reshape(state, [1,4])


	#add a state to our state memory
	def rememberState(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))


	#replay
	def updateNetwork(self, batchSize):

		#tabular version:
		#self.Q[state_old][action] = alpha * (reward + self.gamma * np.max(self.Q[state_new])

		#as we will use batch training, we must gather all the states that we want to pass to the neural net
		x_batch = []

		#we must also gather the labels of these states
		y_batch = []

		#populate the minibatch with a random sample of our past experiences
		minibatch = random.sample(self.memory, min (len(self.memory), batchSize))


		#we will update the q-values of each element of this sample
		for state, action, reward, next_state, done in minibatch:

			#we retrieve the entry that we want to update: (predict() method returns a numpy array)
			target = self.nnModel.predict(state)

			#now we are going to update it

			if done:
				#in this case, there is no next_state, so we update the old_state value only with the reward
				target[0][action] = reward
			else:
				#we use the q-learning equation to update the old state's value
				target[0][action] = reward + self.gamma * np.max(self.nnModel.predict(next_state)[0])

			
			#collecting the sampled states into an array
			x_batch.append(state[0])

			#computing the labels of the states we are going to train the network on
			y_batch.append(target[0])


		#after we finish collecting and updating the sample, we must re-enter them in the network. We do so by training the network on this data.
		self.nnModel.fit(np.array(x_batch), np.array(y_batch), batch_size=batchSize, verbose=0)

		#after updating our network, we should also update the epsilon
		if self.epsilon > self.minEpsilon:
			epsilon*=self.epsilonDecay


	#returns the action we should take when we are in a given state
	def getAction(self, state, epsilon):

		#we test whether a random value is smaller than our threshold epsilon. if so, return a random action:
		if np.random.random() <= epsilon:
			return self.env.action_space.sample()
		else:
			#we use our neural network to predict the value of the possible actions from this state and return the one with the highest value
			return np.argmax(self.nnModel.predict(state))



	def get_epsilon(self, t):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) *self.epsilonDecay)))


    def run(self):

    	scores = deque(maxlen=100)


    	for e in range(self.nEpisodes):

    		state = self.reshapeState(self.env.reset())
    		done = False
    		i=0

    		while not done:
    			










