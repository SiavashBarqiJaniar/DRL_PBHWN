import numpy as np
from numpy import *

class ENVIRONMENT(object):
    """docstring for ENVIRONMENT"""
    def __init__(self,
				 state_size = 10,
				 ):
        super(ENVIRONMENT, self).__init__()
        self.state_size = state_size
        self.action_space = ['w', 't'] # w: wait t: transmit
        self.n_actions = len(self.action_space)
        self.n_nodes = 3
    
    def reset(self):
        init_state = np.zeros(self.state_size, int)
        return init_state

    def sigmoid(self, x):
        y = 20/(1 + exp(-.03*x)) - 10
        return y
    
    def step(self, action, agent):
        n = len(action)
        agent_reward = [0 for j in range(n)]
        ack = [0 for j in range(n)]
        reward = [0 for j in range(n)]
        cap = 1
        cap -= sum(action)
        return ack, reward, agent_reward, cap

