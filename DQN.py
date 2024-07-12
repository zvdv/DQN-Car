# Misc imports
import numpy as np
import random as rnd

# These are your other files.
from buffer import ReplayBuffer
from model import Model
from params import BUFFER_BATCH_SIZE

import torch

class DQNAgent():
    def __init__(self, input_dims, output_dims):
            self.output_dims = output_dims
            self.input_dims = input_dims
            #self.observation_space = observation_space

            self.model = Model(input_dims, output_dims)
            self.target_model = Model(input_dims, output_dims)

            #self.replay_memory =             

            #self.optimizer =

            # this is only important if you're using pytorch. it speeds things up. alot. 
            #self.device = 

    # Method for predicting an action 
    def get_action(self, state) -> int:
        ''' 
        Get action function call.
        Ideally your state is processed by your target network. 

        Your state can be inputted into this function as an array/tuple, in which case
        needs to be turned into a tensor before being inputted into your network.

        or it can be inputted into this function as a tensor already. 
        mostly fashion. do what you please.
        '''
        state_tensor = torch.tensor(state)
        result = self.model(state_tensor)
        action = torch.argmax(result)

        return action
    
    def learn() -> float:
        ''' 
        This function will be the source of 90% of your problems at the
        start. this is where the magic happens. it's also where the tears happen.

        ask questions. please.

        I'll leave a lot more things up here to make it less painful.

        it returns a tuple in case you want to keep track of your losses (you do)
        '''
        loss = 0
        # We just pass through the learn function if the batch size has not been reached. 
        if ReplayBuffer.__len__() < BUFFER_BATCH_SIZE:
            #print("Returning!")
            return

        #print(BUFFER_BATCH_SIZE)

        state = []
        action = []
        reward = []
        next_state = []
        for i in range(BUFFER_BATCH_SIZE):
            #print(i)
            s, a, r, n = ReplayBuffer.collect_memory()
            # append to lists above probably
            state.append(s)
            action.append(a)
            reward.append(r)
            next_state.append(n)

        #print('Shape of state: ', len(state), ' by ', len(state[0])) # 30x4
        #print("Shape of action: ", len(action)) # 30x1
        #print("Len of reward: ", len(reward)) # 30x1
        #print("Shape of next state: ", len(next_state), " by ", len(next_state[0])) # 30x4

        # Convert list of tensors to tensor.
        state = torch.tensor(state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        next_state = torch.tensor(next_state)
        # state = torch.stack(state, dim=0)
        # action = torch.stack(action, dim=0)
        # reward = torch.stack(reward, dim=0)
        # next_state = torch.stack(reward, dim=0)
        
        '''
        State: [cart position, cart velocity, pole angle, pole angular velocity]
        '''

        # One hot encoding our actions. 

        # Find our predictions
        
        # Get the training model assessed Q value of the current turn. 

        # get max value
        max = np.max(DQNAgent.get_action(next_state))

        # Calculate our target
        targ = DQNAgent.get_action(state)

        difference = reward + max - targ

        # Calculate MSE Loss
        loss = torch.nn.MSELoss(targ, reward+max)

        # backward pass

        # self.update_target_counter += 1

        #if self.update_target_counter % TARGET_UPDATE == 0:
            # update

        return loss 

    def save(self, save_to_path: str) -> None:
        # if pytorch
        torch.save(self.target_model.state_dict(), save_to_path)
        pass

    def load(self, load_path: str) -> None:

        # if tensorflow
        #loaded_target = tf.keras.models.load_model(load_path)
        #loaded_model = tf.keras.models.load_model(load_path)

        # if pytorch
        self.target_model.load_state_dict(torch.load(load_path))
        self.model.load_state_dict(torch.load(load_path))

        pass




if __name__ == "__main__":
    '''
    For those unfamiliar with this format, this is so that if you want to run this file
    instead of the main.py file to test this file specifically, everything in this block will be run.
    So, if you had a print statement outside of this block and called functions or classes,
    they will be ignored. 
    '''
    # input_dims = 4
    # output_dims = 2
    # buffer = DQNAgent(input_dims, output_dims)
    DQNAgent.learn()
    print('dqn agent')

