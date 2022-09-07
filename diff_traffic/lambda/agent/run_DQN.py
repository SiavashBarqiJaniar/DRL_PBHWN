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

    number_of_DRLs = NN
    start = time.time()
    ii = 0
    thr = []
    lamb = 401
    for l in range(50):
        state = {}
        reward_list = []
        state_store = [[] for i in range(NN)]
        agent_action = [0 for i in range(NN)]
        for j in range(NN):
            state[j] = env.reset()
        v = [[] for i in range(NN)]

        agent = []
        for j in range(NN):
            agent.append(Agent(60, test, NN, lamb))#40
        lamb -= 8

        for i in range(max_iter):
            for j in range(number_of_DRLs):
                state_store[j].append(state[j])
                agent_action[j], values = dqn_agent.choose_action(state[j]) # argmax(Q(shape=(1,2))) = 0 || 1
                agent[j].action_list.append(agent_action[j])
                v[j].append([ values[0], values[1] ])
            
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
            
            for j in range(number_of_DRLs):
                if max(agent[j].pckt_ratio) > 1:
                    opp = 1
                elif max(agent[j].pckt_ratio) == 1:
                    opp = 2
                elif max(agent[j].pckt_ratio) == 0 and agent[j].real_packets == 0:
                    opp = -1
                else:
                    opp = 0
                next_state = np.concatenate([ state[j][4:], [agent_action[j], cap, ack[j], opp] ])

                dqn_agent.store_transition(state[j], agent_action[j], reward[j], next_state) # SO IMPORTANT!!!! stores the trajectory

                state[j] = next_state
        
        for j in range(number_of_DRLs):
            f = open('arrival_times/agent' + str(j) + "_arrival_times" + str(l) + ".txt" , "w")
            sss = ''
            for ii in agent[j].arrival_time_history:
                sss += str(ii) + '\n'
            f.write(sss)
            f.close()
            
        tempp = 0
        for tt in range(max_iter):
            tempp += sum(reward_list[tt])
        thr.append(tempp/max_iter)
        
    print('\r\n       End of the Game!!!!!!!')
    print(thr)
    plt.plot(thr)
    plt.show()

    for j in range(number_of_DRLs):
        with open('agent_throughput.txt', 'w') as f:
            for listitem in thr:
                f.write('%s\n' % listitem)
    
    print('-----------------------------')
    print('Time elapsed:', time.time()-start)

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
    p_coll = 0
    for i in range(max_iter):
        temp = 1
        for j in range(number_of_DRLs):
            temp *= (1-agent[j].collisions[i])
        p_coll += 1 - temp
    agent0 /= 1000
    agent1 /= 1000
    agent2 /= 1000
    p_coll /= 1000
    p_idle = sum(idle)/1000 - p_coll
    print(agent0, agent1, agent2, p_idle, p_coll)
    print(agent0 + agent1 + agent2 + p_idle + p_coll)
    names = ['agent 0', 'agent 1', 'agent 2', 'idle', 'collision']
    values = [agent0, agent1, agent2, p_idle, p_coll]
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
    test = True

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

    main(max_iter=20000)