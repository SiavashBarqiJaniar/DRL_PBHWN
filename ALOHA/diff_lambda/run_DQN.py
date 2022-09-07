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
    occ_list = []
    number_of_DRLs = NN
    start = time.time()
    state_store = [[] for i in range(NN)]
    agent_action = [0 for i in range(NN)]
    for j in range(NN):
        state[j] = env.reset()
        #state[j][-1] = j
    v = [[] for i in range(NN)]
    output = [[] for i in range(NN)]
    ii = 0
    for i in range(max_iter):
        if i > 0:
            ii = (i-1)%10000
        for j in range(number_of_DRLs):
            output[j].append([])
        for j in range(number_of_DRLs):
            state_store[j].append(state[j])
            if np.random.random() <= 0.33:
                agent_action[j] = 1
            else:
                agent_action[j] = 0
            values = [999, 999]
            agent[j].action_list.append(agent_action[j])
            v[j].append([ values[0], values[1] ])
            
        p = []
        for j in range(number_of_DRLs):
            p.append(agent[j].packets)

        for j in range(number_of_DRLs):
            output[j][ii].append(str(state[j][-4:]))
            output[j][ii].append(str(v[j][i]))
            output[j][ii].append(str(agent_action))
            xx = agent[j].get_opp_buffer()
            output[j][ii].append(str(xx))
            output[j][ii].append(str(p))
        
        #############
        #    Step
        #############
        ack, reward, agent_reward, cap = env.step(agent_action, agent)
        
        for j in range(number_of_DRLs):
            output[j][ii].append(str(reward))

        reward_list.append(agent_reward)
        occ = []
        for j in range(number_of_DRLs):
            if agent_action[j] == 1 and cap == 0:
                occ.append(1)
            else:
                occ.append(0)
        occ_list.append(occ)

        #    Tx
        #############
        for j in range(number_of_DRLs):
            agent[j].tx(agent_action[j], cap, i, j, dqn_agent.epsilon)
        
        opponent_buffer = [False for x in range(number_of_DRLs-1)]
        for j in range(number_of_DRLs):
            if max(agent[j].pckt_ratio) >= 1:
                opp = 1
            else:
                opp = 0
            #opp = [max(x, 2) for x in agent[j].pckt_ratio]
            next_state = np.concatenate([ state[j][4:], [agent_action[j], cap, ack[j], opp] ])

            dqn_agent.store_transition(state[j], agent_action[j], reward[j], next_state) # SO IMPORTANT!!!! stores the trajectory

            state[j] = next_state

        
        """if i == 1000:
            print(next_state)
            exit()"""
        
        if i%10000 == 0:
            print(i)
            
        if i%10000 == 0:
            for j in range(number_of_DRLs):
                f = open('hist/' + str(int(i/10000)) + 'agent' + str(j) + ".txt" , "w")
                for t, k in enumerate(output[j]): # s v a ratio p r
                    string = 'time'+str(t+i-10000)+' :    '+str(k[0])+' --> '+str(k[1])
                    string += ' --> '+str(k[2]) + ' --> r: '+str(k[5])
                    string += '\r\n           true already packets: '+str(k[4])
                    string += '\r\n           opp_buffers: '+str(output[j][t][3]) + '\r\n'
                    f.write(string)
                    #print(output[j][t][3])
                f.close()
                with open('rewards/agent' + str(j) + '_M20.txt', 'w') as my_agent:
                    for k in reward_list:
                        my_agent.write(str(k[j]) + '   ')
                with open('rewards/aloha_occ' + str(j) + '_M20.txt', 'w') as my_agent:
                    for k in occ_list:
                        my_agent.write(str(k[j]) + '   ')
                with open('rewards/agent_action' + str(j) + '_M20.txt', 'w') as my_agent2:
                    for k in agent[j].action_list:
                        my_agent2.write(str(k) + '   ')
                with open('rewards/agent_buffer' + str(j) + '_M20.txt', 'w') as my_agent3:
                    for k in agent[j].buffer_history:
                        my_agent3.write(str(k) + '   ')
                with open('rewards/agent_collision' + str(j) + '_M20.txt', 'w') as my_agent4:
                    for k in agent[j].collisions:
                        my_agent4.write(str(k) + '   ')
            output = [[] for i in range(NN)]
            
    print('\r\n       End of Learning!!!!!!!')

    for j in range(number_of_DRLs):
        f = open('hist/' + str(10) + 'agent' + str(j) + ".txt" , "w")
        for t, k in enumerate(output[j]): # s v a ratio p r
            string = 'time'+str(t+90000)+' :    '+str(k[0])+' --> '+str(k[1])
            string += ' --> '+str(k[2]) + ' --> r: '+str(k[5])
            string += '\r\n           true already packets: '+str(k[4])
            string += '\r\n           opp_buffers: '+str(output[j][t][3]) + '\r\n'
            f.write(string)
        f.close()
        with open('rewards/agent' + str(j) + '_M20.txt', 'w') as my_agent:
            for i in reward_list:
                my_agent.write(str(i[j]) + '   ')
        with open('rewards/aloha_occ' + str(j) + '_M20.txt', 'w') as my_agent:
            for k in occ_list:
                my_agent.write(str(k[j]) + '   ')
        with open('rewards/agent_action' + str(j) + '_M20.txt', 'w') as my_agent2:
            for i in agent[j].action_list:
                my_agent2.write(str(i) + '   ')
        with open('rewards/agent_buffer' + str(j) + '_M20.txt', 'w') as my_agent3:
            for i in agent[j].buffer_history:
                my_agent3.write(str(i) + '   ')
        with open('rewards/agent_collision' + str(j) + '_M20.txt', 'w') as my_agent4:
            for i in agent[j].collisions:
                my_agent4.write(str(i) + '   ')
    
    print('-----------------------------')
    for j in range(number_of_DRLs):
        print('average agent'+str(j)+' reward: {}'.format(np.mean(agent[j].reward_list[-2000:][j])))
    print('average total reward: {}'.format(np.mean(agent[j].reward_list[-2000:])))
    print('Time elapsed:', time.time()-start)
    for j in range(number_of_DRLs):
        print(str(j) + ' : %d' %(agent[j].packets))
    if not test:
        dqn_agent.model.save('DQN.h5')

if __name__ == "__main__":
    NN = 3
    test = True
    agent = []
    agent.append(Agent(60, test, NN, 800))
    for j in range(NN-1):
        agent.append(Agent(60, test, NN, 200))#60
    #agent.append(Agent(140, test))
    #agent.append(Agent(60, test))
    #agent.append(Agent(80, test))

    env = ENVIRONMENT(state_size=4*1)
    
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