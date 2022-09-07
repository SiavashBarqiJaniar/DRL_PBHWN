import numpy as np

states = []
values = []
actions = []
for j in range(3):
    states.append(np.loadtxt('rewards/agent_state'+str(j)+'_M20.txt'))
    values.append(np.loadtxt('rewards/agent_value'+str(j)+'_M20.txt'))
    actions.append(np.loadtxt('rewards/agent_action'+str(j)+'_M20.txt'))

t = input('enter time: ')
n = input('enter agent: ')
for i in range(10):
    print(str(i) + ' : ' + str(states[n][t+i]) + ' --> Qs: ' + str(values[n][t+i]) + ' --> action: ' + str(actions[n][t+i]))