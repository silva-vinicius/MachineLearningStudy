'''
Reinforcement Learning induction:

The agent takes random actions that are biased by the neural net weights and the input.

Pole balancing problem:

	input: [x, θ, dx, dθ]

	x -> position of the cart on the track
	θ -> angle of the pole with the vertical
	dx -> cart velocity
	dθ -> rate of change of the pole angle

	if we receive the pole falls or hits the track boundary, we get
    a failure signal, we reset the cart position, but not the controller's 
    memory

'''

import gym
import numpy as np
from time import sleep


#returns a list containing [[array of weights], bias]
def generate_random_policy():
	return(np.random.uniform(-1,1, size=4), np.random.uniform(-1,1))



def policy_to_action(env, policy, obs):

	if((np.dot(policy[0], obs) + policy[1]) > 0):
		return 1 #move the cart to the right
	else:
		return 0 #move the cart to the left


#runs an episode and returns the accumulated reward
def run_episode(env, policy, max_iterations=2000, render=False, breakOnDone=True):

	obs = env.reset()

	if(render):
		env.render()
		

	total_reward = 0

	for i in range(max_iterations):

		if render:
			#render the scene
			env.render()
			sleep(0.017)

		#we define an action
		action = policy_to_action(env, policy, obs)

		#we perform the action and collect the environment's feedback
		obs, reward, done, info = env.step(action)

		if done and breakOnDone:
			break

		total_reward += reward

	return total_reward


if __name__ == '__main__':
	
	noise_scaling = 0.1
	env = gym.make('CartPole-v0')
	policy = generate_random_policy()[0]

	bestreward = 0


	for i in range(10000):

		print('Episode ' +  repr(i) + ':\n')
		rand_policy = generate_random_policy()
		newpolicy = (policy[0] + (rand_policy[0])*noise_scaling , rand_policy[1]*noise_scaling)
		reward = 0

		reward = run_episode(env, newpolicy)


		if reward > bestreward:

			bestreward = reward
			policy = newpolicy
			if reward >= 200:
				print('Problem solved!')
				break

		print("Current reward/ Best reward " + repr(reward) + '/' + repr(bestreward) + "\n")
		print("Best policy: " + repr(policy) + "\n")
		print("Current policy " + repr(newpolicy) + "\n")


	run_episode(env, policy, render=True, breakOnDone=False)

