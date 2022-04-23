import gym
import time

# init the gym environment
e = gym.make('CartPole-v0')
# There are four class members inside this class
# action_space, observation_space, reset(), step() and render()
# reset the env, and return the observation 
print(e.action_space)
print(e.observation_space)


