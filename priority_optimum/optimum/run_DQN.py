from environment import ENVIRONMENT
from agent import Agent

from math import *
import numpy as np
import matplotlib.pyplot as plt
import time

def main(max_iter):
    print('------------------------------------------')
    print('---------- Start processing ... ----------')
    print('------------------------------------------')

    reward_list = []
    number_of_DRLs = NN
    output = ''
    start = time.time()
    output = [[] for i in range(NN)]
    ii = 0
    for i in range(max_iter):
        priority = 0
        agent_action = [0 for i in range(NN)]
        if i > 0:
            ii = (i-1)%10000
        for j in range(number_of_DRLs):
            output[j].append([])
        for j in range(number_of_DRLs):
            if agent[j].threes > 0:
                priority = 3
            elif agent[j].twos > 0 and priority <= 2:
                priority = 2
            elif agent[j].ones > 0 and priority <= 1:
                priority = 1
        maxx = 0
        tx_agent = -1
        for j in range(number_of_DRLs):
            if priority == 3:
                if agent[j].threes > maxx:
                    maxx = agent[j].threes
                    tx_agent = j
            elif priority == 2:
                if agent[j].twos > maxx:
                    maxx = agent[j].twos
                    tx_agent = j
            elif priority == 1:
                if agent[j].ones > maxx:
                    maxx = agent[j].ones
                    tx_agent = j
        if tx_agent > -1:
            agent_action[tx_agent] = 1
        for j in range(number_of_DRLs):
            agent[j].action_list.append(agent_action[j])
            
        p = []
        for j in range(number_of_DRLs):
            p.append(agent[j].packets)

        for j in range(number_of_DRLs):
            output[j][ii].append(str(agent_action))
            xx = agent[j].get_opp_buffer()
            output[j][ii].append(str(xx))
            output[j][ii].append(str(p))
        
        #############
        #    Step
        #############
        ack, reward, agent_reward, cap = env.step(agent_action, agent)

        reward_list.append(agent_action)

        #############
        #    Tx
        #############
        for j in range(number_of_DRLs):
            agent[j].tx(agent_action[j], cap, i, j)
        
        """if i == 1000:
            print(next_state)
            exit()"""
        
        if i%10000 == 0:
            print(i)
            
        if i%10000 == 0:
            for j in range(number_of_DRLs):
                f = open('hist/' + str(int(i/10000)) + 'agent' + str(j) + ".txt" , "w")
                for t, k in enumerate(output[j]): # s v a ratio p r
                    string = 'time'+str(t+i-10000)+' :    '+' --> '+str(k[0])
                    string += '\r\n           true already packets: '+str(k[2])
                    string += '\r\n           opp_buffers: '+str(output[j][t][1]) + '\r\n'
                    f.write(string)
                    #print(output[j][t][3])
                f.close()
                with open('rewards/' + str(j) + '_M20.txt', 'w') as my_agent:
                    for k in reward_list:
                        my_agent.write(str(k[j]) + '   ')
                with open('rewards/action' + str(j) + '_M20.txt', 'w') as my_agent2:
                    for k in agent[j].action_list:
                        my_agent2.write(str(k) + '   ')
                with open('rewards/threes_buffer' + str(j) + '_M20.txt', 'w') as my_agent3:
                    for k in agent[j].threes_history:
                        my_agent3.write(str(k) + '   ')
                with open('rewards/twos_buffer' + str(j) + '_M20.txt', 'w') as my_agent3:
                    for k in agent[j].twos_history:
                        my_agent3.write(str(k) + '   ')
                with open('rewards/ones_buffer' + str(j) + '_M20.txt', 'w') as my_agent3:
                    for k in agent[j].ones_history:
                        my_agent3.write(str(k) + '   ')
            output = [[] for i in range(NN)]
            
    print('\r\n       End of Learning!!!!!!!')

    for j in range(number_of_DRLs):
        f = open('hist/' + str(10) + 'agent' + str(j) + ".txt" , "w")
        for t, k in enumerate(output[j]): # s v a ratio p r
            string = 'time'+str(t+90000)+' :    '+' --> '+str(k[0])
            string += '\r\n           true already packets: '+str(k[2])
            string += '\r\n           opp_buffers: '+str(output[j][t][1]) + '\r\n'
            f.write(string)
        f.close()
        with open('rewards/' + str(j) + '_M20.txt', 'w') as my_agent:
            for i in reward_list:
                my_agent.write(str(i[j]) + '   ')
        with open('rewards/action' + str(j) + '_M20.txt', 'w') as my_agent2:
            for i in agent[j].action_list:
                my_agent2.write(str(i) + '   ')
        with open('rewards/threes_buffer' + str(j) + '_M20.txt', 'w') as my_agent3:
            for k in agent[j].threes_history:
                my_agent3.write(str(k) + '   ')
        with open('rewards/twos_buffer' + str(j) + '_M20.txt', 'w') as my_agent3:
            for k in agent[j].twos_history:
                my_agent3.write(str(k) + '   ')
        with open('rewards/ones_buffer' + str(j) + '_M20.txt', 'w') as my_agent3:
            for k in agent[j].ones_history:
                my_agent3.write(str(k) + '   ')
    
    print('-----------------------------')
    for j in range(number_of_DRLs):
        print('average agent'+str(j)+' reward: {}'.format(np.mean(agent[j].reward_list[-2000:][j])))
    print('average total reward: {}'.format(np.mean(agent[j].reward_list[-2000:])))
    print('Time elapsed:', time.time()-start)
    for j in range(number_of_DRLs):
        print(str(j) + ' : %d' %(agent[j].threes, agent[j].twos, agent[j].ones))

if __name__ == "__main__":
    NN = 3
    agent = []
    for j in range(NN):
        agent.append(Agent(NN, 300, 10, 10, 10))
    #agent.append(Agent(140, test))
    #agent.append(Agent(60, test))
    #agent.append(Agent(80, test))

    env = ENVIRONMENT(state_size=4*1)

    main(max_iter=100000)