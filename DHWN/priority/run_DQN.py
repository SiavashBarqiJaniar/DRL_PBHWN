from environment import ENVIRONMENT
from agent import Agent
from DQN_brain import DQN

from math import *
import numpy as np
import matplotlib.pyplot as plt
import time

def main(max_iter):
    print('------------------------------------------')
    print('---------- Start processing ... ----------')
    print('------------------------------------------')

    state = {}
    reward_list = []
    number_of_DRLs = NN
    packs = {}
    output = ''
    start = time.time()
    state_store = [[] for i in range(NN)]
    agent_action = [0 for i in range(NN)]
    for j in range(NN):
        state[j] = env.reset()
    for i in range(max_iter):
        #Intelligent
        state_store[0].append(state[0])
        agent_action[0], _ = dqn_agent.choose_action(state[0]) # argmax(Q(shape=(1,2))) = 0 || 1
        #Slotted-ALOHA
        if np.random.random() <= 0.4:
            agent_action[1] = 1
        else:
            agent_action[1] = 0
        #TDMA
        if i%10 == 1 or i%10 == 7 or i%10 == 8:
            agent_action[2] = 1
        else:
            agent_action[2] = 0
        for j in range(number_of_DRLs):
            agent[j].action_list.append(agent_action[j])
        
        #############
        #    Step
        #############
        ack, reward, agent_reward, cap = env.step(agent_action, agent)

        reward_list.append(agent_reward)

        #############
        #    Tx
        #############
        for j in range(number_of_DRLs):
            agent[j].tx(agent_action[j], cap, i, j, dqn_agent.epsilon)
        
        opponent_buffer = [False for x in range(number_of_DRLs-1)]
        for j in range(number_of_DRLs):
            if max(agent[j].three_ratio) > 1:
                opp = 1
            elif max(agent[j].three_ratio) == 1:
                opp = 2
            elif max(agent[j].three_ratio) < 1 and max(agent[j].three_ratio) > 0:
                opp = 0
            elif max(agent[j].three_ratio) == 0 and agent[j].threes > 0:
                opp = 0
            elif max(agent[j].three_ratio) == 0 and agent[j].threes == 0:
                if max(agent[j].two_ratio) > 1:
                    opp = 1
                elif max(agent[j].two_ratio) == 1:
                    opp = 2
                elif max(agent[j].two_ratio) < 1 and max(agent[j].two_ratio) > 0:
                    opp = 0
                elif max(agent[j].two_ratio) == 0 and agent[j].twos > 0:
                    opp = 0
                elif max(agent[j].two_ratio) == 0 and agent[j].twos == 0:
                    if max(agent[j].one_ratio) > 1:
                        opp = 1
                    elif max(agent[j].one_ratio) == 1:
                        opp = 2
                    elif max(agent[j].one_ratio) < 1 and max(agent[j].one_ratio) > 0:
                        opp = 0
                    elif max(agent[j].ones_ratio) == 0 and agent[j].ones > 0:
                        opp = 0
                    elif max(agent[j].ones_ratio) == 0 and agent[j].ones == 0:
                        opp = -1
            next_state = np.concatenate([ state[j][4:], [agent_action[j], cap, ack[j], opp] ])

            dqn_agent.store_transition(state[j], agent_action[j], reward[j], next_state) # SO IMPORTANT!!!! stores the trajectory

            state[j] = next_state

        if not test:
            if i > 200:
                dqn_agent.learn()
    
        if i%10000 == 0:
            print(i)
            
    print('\r\n       End of Learning!!!!!!!')

    for j in range(number_of_DRLs):
        with open('rewards/agent' + str(j) + '_M20.txt', 'w') as my_agent:
            for i in reward_list:
                my_agent.write(str(i[j]) + '   ')
        with open('rewards/agent_action' + str(j) + '_M20.txt', 'w') as my_agent2:
            for i in agent[j].action_list:
                my_agent2.write(str(i) + '   ')
        with open('rewards/agent_threes_buffer' + str(j) + '_M20.txt', 'w') as my_agent3:
            for k in agent[j].threes_history:
                my_agent3.write(str(k) + '   ')
        with open('rewards/agent_twos_buffer' + str(j) + '_M20.txt', 'w') as my_agent3:
            for k in agent[j].twos_history:
                my_agent3.write(str(k) + '   ')
        with open('rewards/agent_ones_buffer' + str(j) + '_M20.txt', 'w') as my_agent3:
            for k in agent[j].ones_history:
                my_agent3.write(str(k) + '   ')
        with open('rewards/agent_collision' + str(j) + '_M20.txt', 'w') as my_agent4:
            for i in agent[j].collisions:
                my_agent4.write(str(i) + '   ')
    
    print('-----------------------------')

    if not test:
        dqn_agent.model.save('DQN.h5')

if __name__ == "__main__":
    NN = 3
    test = False
    agent = []
    for j in range(NN):
        agent.append(Agent(test, NN, 300, 10, 10, 10))

    env = ENVIRONMENT(state_size=4*2)
    
    dqn_agent = DQN(env.state_size,
                    env.n_actions,
                    env.n_nodes,
                    memory_size=NN*1500,
                    replace_target_iter=200,
                    batch_size=NN*32,
                    learning_rate=0.01,
                    gamma=0.9,
                    epsilon=0.6,
                    epsilon_min=0.005,
                    epsilon_decay=0.99999,
                    test = test,
                    )

    main(max_iter=100000)