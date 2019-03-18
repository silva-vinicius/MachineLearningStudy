from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import numpy as np
import gym
import math
import random
from collections import deque
import os.path

class DeepQCartSolver():

    def __init__(self, nEpisodes=2000, minScoreToWin=195, gamma=0.5, epsilon=1.0, minEpsilon=0.01, epsilonDecay=0.995,
                 alpha=0.01, alphaDecay=0.01, batchSize=64):

        self.env = gym.make('CartPole-v0')
        self.nEpisodes = nEpisodes
        self.minScoreToWin = minScoreToWin
        self.memory = deque(maxlen=50000)  # This dataset will feed our neural network with data
        self.gamma = gamma # This quantifies the difference in importance between immediate rewards and future rewards
        self.epsilon = epsilon # controls the probability of performing a random action
        self.minEpsilon = minEpsilon
        self.epsilonDecay = epsilonDecay
        self.alpha = alpha # controls how fast we optimize using gradient descent
        self.alphaDecay = alphaDecay
        self.batchSize = batchSize

        self.nnModel = self.initNNModel()

    # layerNeurons -> input_nodes, hidden_nodes, hidden_nodes, output_node
    # activationFunc -> hidden_nodes activation, hidden_nodes activation, output_activation
    def initNNModel(self, layerNeurons=[4, 24, 48, 2], activationFunc=['relu', 'relu', 'linear'], lossFunc='mse'):

        neuralNet = Sequential()

        for i in range(len(layerNeurons) - 1):

            if i == 0:

                # adds the input layer and the first hidden layer
                print(i)
                neuralNet.add(Dense(layerNeurons[i + 1], input_dim=layerNeurons[i], activation=activationFunc[i]))

            else:
                if i != (len(layerNeurons) - 2):

                    # adds all the other hidden layers
                    neuralNet.add(Dense(layerNeurons[i + 1], activation=activationFunc[i]))

                else:
                    # adds the last layer
                    neuralNet.add(Dense(layerNeurons[i+1], activation=activationFunc[i]))

        neuralNet.compile(loss=lossFunc, optimizer=Adam(lr=self.alpha, decay=self.alphaDecay))

        if(os.path.isfile("weights.txt")):
            neuralNet.load_weights("weights.txt")

        return neuralNet

    # transforms a python list into a numpy array so that keras can understand it
    def reshapeState(self, state):
        return np.reshape(state, [1, 4])

    # add a state to our state memory
    def rememberState(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # replay
    def updateNetwork(self, batchSize):

        # tabular version:
        # self.Q[state_old][action] = alpha * (reward + self.gamma * np.max(self.Q[state_new])

        # as we will use batch training, we must gather all the states that we want to pass to the neural net
        # x_batch is a list of states (list of arrays[4])
        x_batch = []

        # we must also gather the labels of these states
        # y_batch is a list of q-values (list of arrays[2])
        y_batch = []

        # populate the minibatch with a random sample of our past experiences
        minibatch = random.sample(list(self.memory), min(len(self.memory), batchSize))
        # obs.: we use the whole memory if the specified batch size is greater than the actual available memory

        # we will update the q-values of each element of this sample
        for state, action, reward, next_state, done in minibatch:

            # state shape [[x,y,z,w] ]

            #We are querying the NN for the value we want to update
            # we retrieve the entry that we want to update: (predict() method returns a numpy array)
            target = self.nnModel.predict(self.reshapeState(state))
            # target shape [[x,y] ] -> x is the q-value of action 0 and y is the q-value for action 1

            # now we are going to update it

            if done:
                # in this case, there is no next_state, so we update the old_state value only with the reward
                target[0][action] = reward
            else:
                # we use the q-learning equation to update the old state's value
                target[0][action] = reward + self.gamma * np.max(self.nnModel.predict(self.reshapeState(next_state)))

            # collecting the sampled states into an array
            # state[0] shape -> [x,y,z,w]
            x_batch.append(state[0])

            # computing the labels of the states we are going to train the network on
            y_batch.append(target[0])

        # after we finish collecting and updating the sample, we must re-enter them in the network. We do so by training the network on this data.
        self.nnModel.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

        # after updating our network, we should also update the epsilon
        if self.epsilon > self.minEpsilon:
            self.epsilon *= self.epsilonDecay

    # returns the action we should take when we are in a given state
    def getAction(self, state, epsilon): #Epsilon is the threshold for random actions

        # we test whether a random value is smaller than our threshold epsilon. if so, return a random action:
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            # we use our neural network to predict the value of the possible actions from this state and return the one with the highest value
            return np.argmax(self.nnModel.predict(state)) #the predict method returns the estimation of the values of each possible state

    def getEpsilon(self, t):

        return max(self.minEpsilon, min(1, (1.0 - (math.log10((t + 1) * self.epsilonDecay)))))

    def run(self):
        scores = deque(maxlen=100)

        # the code will run as many episodes as we want -> nEpisodes = 2000
        for e in range(self.nEpisodes):
            # #we read the initial state from the environment
            state = self.reshapeState(self.env.reset())

            done = False
            i = 0


            while not done:
                self.env.render()

                # selecting an action to take given the state we are in
                action = self.getAction(state, self.getEpsilon(e))

                # we take the selected action and read the environment again so we can see how it changed
                next_state, reward, done, _ = self.env.step(action)

                # we add our move to the agent's memory -- We will iteratively gather data to train our network
                self.rememberState(state, action, reward, self.reshapeState(next_state), done)

                # we assume that the current state is last step's resulting state
                state = self.reshapeState(next_state)
                i += 1

            scores.append(i)
            avg_score = np.mean(scores)

            if avg_score > self.minScoreToWin:
                print('Ran {} episodes. Solved after {} trials'.format(e, e - 100))
                self.nnModel.save_weights("weights.txt",True)
                return e - 100

            if (e % 100 == 0):
                print('Episode {} - Average survival time over last 100 episodes was {} ticks'.format(e, avg_score))

            # we update the neural net after each episode
            self.updateNetwork(self.batchSize)

            print("Avg score: {} / Last score: {} \nEpisode: {} Epsilon: {}".format(avg_score, i, e, self.epsilon))

        print('Algorithm was unable to solve after {} epsiodes.'.format(e))

        return e


if __name__ == '__main__':
    agent = DeepQCartSolver()
    agent.run()