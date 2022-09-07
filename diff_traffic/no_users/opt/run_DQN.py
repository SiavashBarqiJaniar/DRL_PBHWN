from agent import Agent

from math import *
import numpy as np
import matplotlib.pyplot as plt
import time

def main(max_iter):
    print('------------------------------------------')
    print('---------- Start processing ... ----------')
    print('------------------------------------------')

    start = time.time()
    thr = []
    lamb = 300
    NN = 1
    for l in range(40):
        NN += 1
        number_of_DRLs = NN
        reward_list = []
        #############
        #    Agents
        #############
        agent = []
        for j in range(NN):
            arr_times = []
            with open('arrival_times/agent' + str(j) + "_arrival_times" + str(l) + ".txt", 'r') as f:
                for line in f:
                    current_place = line[:-1]
                    arr_times.append(int(current_place))
            agent.append(Agent(60, NN, lamb, arr_times))

        for i in range(max_iter):
            agent_action = [0 for i in range(NN)]
            maxx = 0
            tx_agent = -1
            for j in range(number_of_DRLs):
                if agent[j].real_packets > maxx:
                    maxx = agent[j].real_packets
                    tx_agent = j
            if tx_agent > -1:
                agent_action[tx_agent] = 1
            for j in range(number_of_DRLs):
                agent[j].action_list.append(agent_action[j])
                
            #############
            #    Step
            #############

            reward_list.append(agent_action)
            agent[j].reward_list.append(agent_action)

            #############
            #    Tx
            #############
            for j in range(number_of_DRLs):
                agent[j].tx(agent_action[j], i, j)

        tempp = 0
        for tt in range(max_iter):
            tempp += sum(reward_list[tt])
        thr.append(tempp/max_iter)

    
    for j in range(number_of_DRLs):
        with open('opt_throughput.txt', 'w') as f:
            for listitem in thr:
                f.write('%s\n' % listitem)

    print('\r\n       End of Learning!!!!!!!')
    print(thr)
    print('Time elapsed:', time.time()-start)
    plt.plot(thr)
    plt.show()

    idle = []
    agent0 = 0
    agent1 = 0
    agent2 = 0
    for x in reward_list:
        temp = 1
        agent0 += x[0]
        agent1 += x[1]
        agent2 += x[2]
        for y in x:
            temp *= (1-y)
        idle.append(temp)
    agent0 /= 1000
    agent1 /= 1000
    agent2 /= 1000
    p_idle = sum(idle)/1000
    print(agent0, agent1, agent2, p_idle)
    print(agent0 + agent1 + agent2 + p_idle)
    names = ['agent 0', 'agent 1', 'agent 2', 'idle', 'collision']
    values = [agent0, agent1, agent2, p_idle, 0]
    fig, ax = plt.subplots(1, 1, sharex=True)
    fig.subplots_adjust(hspace=0)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    ax.bar(names, values)
    plt.xlabel('Time', fontsize=24)
    plt.ylabel('Channel usage percentage', fontsize=24)
    plt.show()

if __name__ == "__main__":
    NN = 3

    main(max_iter=4000)