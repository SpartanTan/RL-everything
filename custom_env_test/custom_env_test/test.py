import gymnasium as gym
import gym_examples

env = gym.make("gym_examples/GridWorld-v0")
print(env.action_space.sample())