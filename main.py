import numpy as np
import random as rnd
import time 
import matplotlib.pyplot as plt
import math
import gym
import pygame

import torch

# all of the libraries above can be installed with pip
# ex: pip install numpy or pip install torch

from DQN import DQNAgent
from buffer import ReplayBuffer

# Hyperparams
input_dims = 4
output_dims = 2
batch_size = 32 #I picked this pretty randomly
learning_rate = 2e-4
episodes = 0
epsilon = np.linspace(1,0,500)

# Global Constants, change these
MAX_EPISODES = 500
BUFFER_BATCH_SIZE = 30 # chose this number randomly 

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode='human')
    agent = DQNAgent(input_dims, output_dims)

    # Make the main game loop.  

    while episodes < MAX_EPISODES:
        time_step = 0
        rewards = []
        ReplayBuffer.erase_memory()
        observation, info = env.reset()
        done = False

        while not done:
            # Get action, ideally through your agent
            #action = env.action_space.sample()
            action = agent.get_action(observation)
            _action = action # Store tensor version of action (idk yet if this is the right call for what to store in the memory)
            action = action.numpy()
            _observation = observation # Store last observation before it gets updated

            # Take the action and observe the result
            observation, reward, terminated, trunicated, info = env.step(action)
            
            # Accumulate the reward

            # Check if we lost
            if terminated or trunicated:
                done = True

            # Store our memory
            memory = (_observation, _action, reward, observation)
            ReplayBuffer.store_memory(memory)

            # learn?
            DQNAgent.learn()
            time_step += 1

            env.render()

        episodes += 1
        
    # Plot stuff!!
    # - reward
    # - loss per episode
    # - epsilon for fun?

    # TODO: Check if reward normalization makes sense!
    # agent.save()
    env.close()