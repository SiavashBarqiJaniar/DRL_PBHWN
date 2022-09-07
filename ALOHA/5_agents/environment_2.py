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
        rr = [0 for j in range(n)]
        agent_reward = [0 for j in range(n)]
        observation_ = [0 for j in range(n)]
        reward = [0 for j in range(n)]
        cap = 1
        cap -= sum(action)
        collision = [0 for j in range(n)]
        neg_reward = [0 for j in range(n)]
        sccss_reward = [0 for j in range(n)]
        opponent_packets = 0

        if cap == 0:
            for j in range(n):
                if action[j] == 1:
                    opponent_packets = agent[j].real_packets
        
        for j in range(n):
            neg_reward[j] = (sign(agent[j].packets)-1)*sum(agent[j].collisions[-1000:])*50#-sum(agent[j].collisions[-1000:])*1000
            sccss_reward[j] = abs(agent[j].packets/2) # agent[j].real_packets*2#min(agent[j].real_packets/2 , 10)
            
            Buffer_status=.05*self.sigmoid(agent[j].packets)
            ### 12 conditions:

            #action, Colission, by buffer, am I more than the one who transmitted
            if action[j] == 1 and cap == 0 and agent[j].real_packets > 0:
                observation_[j] = 1
                agent_reward[j] = 1
                reward[j] = 1
            elif action[j] == 1 and cap == 0 and agent[j].real_packets == 0:
                observation_[j] = 2
                agent_reward[j] = 1
                reward[j] = -0.5
            elif action[j] == 1 and cap < 0 and agent[j].real_packets > 0:
                observation_[j] = 3
                agent_reward[j] = 0
                reward[j] = Buffer_status #0 
                agent[j].no_collisions += 1
                collision[j] = 1
            elif action[j] == 1 and cap < 0 and agent[j].real_packets == 0:
                observation_[j] = 4
                agent_reward[j] = 0
                agent[j].no_collisions += 1
                collision[j] = 1
                reward[j] = -1

            elif action[j] == 0 and cap == 0 and agent[j].real_packets > 0 and agent[j].packets > opponent_packets:
                observation_[j] = -1
                agent_reward[j] = 0
                reward[j] = -0.5
            elif action[j] == 0 and cap == 0 and agent[j].real_packets > 0 and agent[j].packets <= opponent_packets:
                observation_[j] = -2
                agent_reward[j] = 0
                reward[j] = 2
            elif action[j] == 0 and cap == 0 and agent[j].real_packets == 0 and agent[j].packets >= opponent_packets: ###########
                observation_[j] = -3
                agent_reward[j] = 0
                reward[j] = 1
            elif action[j] == 0 and cap == 0 and agent[j].real_packets == 0 and agent[j].packets < opponent_packets: ############
                observation_[j] = -4
                agent_reward[j] = 1
                if opponent_packets > 0:
                    reward[j] = 2
                else:
                    reward[j] = 1
                
            elif action[j] == 0 and cap < 0 and agent[j].real_packets > 0: #################
                observation_[j] = -5
                agent_reward[j] = 0
                reward[j] = -Buffer_status #0 #-sigmoid
            elif action[j] == 0 and cap < 0 and agent[j].real_packets == 0: #################
                observation_[j] = -6
                agent_reward[j] = 0
                reward[j] = 1
                
            elif action[j] == 0 and cap > 0 and agent[j].real_packets > 0:
                observation_[j] = 10
                agent_reward[j] = 0
                reward[j] = -Buffer_status #-0.5 #-sigmoid
            elif action[j] == 0 and cap > 0 and agent[j].real_packets == 0:
                observation_[j] = 20
                agent_reward[j] = 0
                reward[j] = sccss_reward[j]
                reward[j] = 1
        
        #reward = [self.sigmoid(x) for x in reward]

        for j in range(n):
            agent[j].reward_list.append(rr[j])
            agent[j].clc_throughput()
            agent[j].collisions.append(collision[j])

# =============================================================================
# 		sum : if any of agent_reward's be NEGATIVE, this part of code doesnt make sense
# =============================================================================
        """
        r = max(reward)
        if r == 0:
            r = min(reward)
        reward = [r for x in range(n)]
        #r = 0
        #for j in range(n):
        #    r += (agent[j].throughput)*min(1000,len(agent[j].reward_list))
        #reward = [r for x in range(n)]
		"""
# =============================================================================
# 		sum-log
# =============================================================================
        """
		#r = 0
		#if counter%len(reward_buffer) == 0:
		#	for j in range(n):
		#		r += 10*log10(sum(reward_buffer[:][j]) + 1) # reward*.9^10 = 1 .9^10 ~ .34
        m = []
        for j in range(n):
            m.append(sum(agent[j].reward_list))
        m = [x+.001 for x in range(n)]
        r = 0
        for j in range(n):
            r += (agent_reward[j])/m[j]
        reward = [r for x in range(n)]
		"""
# =============================================================================
# 		otherwise => competitive
# =============================================================================
        """
        reward = agent_reward
		"""
        return observation_, reward, agent_reward

