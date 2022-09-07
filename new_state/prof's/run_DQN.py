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
        #state[j][-1] = j
    v = [[], [], []]
    output = [[], [], []]
    ii = 0
    for i in range(max_iter):
        if i > 0:
            ii = (i-1)%10000
        for j in range(number_of_DRLs):
            output[j].append([])
        if test:
            dqn_agent.ep(i)
        for j in range(number_of_DRLs):
            if False:#agent[j].no_continuous_colls > 2:
                agent_action = [0 for i in range(NN)]
                agent_action[np.random.randint(0, number_of_DRLs)] = 1
                values = [999, 999]
            else:
                state_store[j].append(state[j])
                agent_action[j], values = dqn_agent.choose_action(state[j]) # argmax(Q(shape=(1,2))) = 0 / 1
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

        #############
        #    Tx
        #############
        for j in range(number_of_DRLs):
            if cap < 0 and agent_action[j] == 1:
                agent[j].no_continuous_colls += 1
            else:
                agent[j].no_continuous_colls = 0
                if agent[j].no_continuous_colls < 0:
                    agent[j].no_continuous_colls = 0
            agent[j].tx(agent_action[j], cap, i, j, dqn_agent.epsilon)
        
        opponent_buffer = [False for x in range(number_of_DRLs-1)]
        for j in range(number_of_DRLs):
            """for x in range(number_of_DRLs-1):
                if agent[j].pckt_ratio[x] >= 1:
                    opponent_buffer[x] = 1
                else:
                    opponent_buffer[x] = 0"""
            if max(agent[j].pckt_ratio) >= 2:
                opp = 3
            elif max(agent[j].pckt_ratio) >= 1:
                opp = 2
            elif max(agent[j].pckt_ratio) >= .5:
                opp = 1
            else:
                opp = 0
            #opp = [max(x, 2) for x in agent[j].pckt_ratio]
            next_state = np.concatenate([ state[j][4:], [agent_action[j], cap, ack[j], opp] ])

            dqn_agent.store_transition(state[j], agent_action[j], reward[j], next_state) # SO IMPORTANT!!!! stores the trajectory

            state[j] = next_state

        if not test:
            if i > 200:
                dqn_agent.learn()
        
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
                with open('rewards/agent_action' + str(j) + '_M20.txt', 'w') as my_agent2:
                    for k in agent[j].action_list:
                        my_agent2.write(str(k) + '   ')
                with open('rewards/agent_buffer' + str(j) + '_M20.txt', 'w') as my_agent3:
                    for k in agent[j].buffer_history:
                        my_agent3.write(str(k) + '   ')
                with open('rewards/agent_collision' + str(j) + '_M20.txt', 'w') as my_agent4:
                    for k in agent[j].collisions:
                        my_agent4.write(str(k) + '   ')
                with open('rewards/agent_state' + str(j) + '_M20.txt', 'w') as my_agent5:
                    for k in state_store[j]:
                        k = np.array(k)
                        my_agent5.write(str(k) + '   ')
                with open('rewards/agent_value' + str(j) + '_M20.txt', 'w') as my_agent6:
                    for k in v[j]:
                        my_agent6.write(str(k) + '   ')
            output = [[], [], []]
            
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
        with open('rewards/agent_action' + str(j) + '_M20.txt', 'w') as my_agent2:
            for i in agent[j].action_list:
                my_agent2.write(str(i) + '   ')
        with open('rewards/agent_buffer' + str(j) + '_M20.txt', 'w') as my_agent3:
            for i in agent[j].buffer_history:
                my_agent3.write(str(i) + '   ')
        with open('rewards/agent_collision' + str(j) + '_M20.txt', 'w') as my_agent4:
            for i in agent[j].collisions:
                my_agent4.write(str(i) + '   ')
        with open('rewards/agent_state' + str(j) + '_M20.txt', 'w') as my_agent5:
            for k in state_store[j]:
                k = np.array(k)
                my_agent5.write(str(k) + '   ')
        with open('rewards/agent_value' + str(j) + '_M20.txt', 'w') as my_agent6:
            for k in v[j]:
                my_agent6.write(str(k) + '   ')
    
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
    test = False
    agent = []
    for j in range(NN):
        agent.append(Agent(60, test, NN))
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