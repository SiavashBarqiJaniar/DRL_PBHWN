import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd

### calculate throughput
def cal_throughput(max_iter, N, reward):
	temp_sum = 0
	throughput = np.zeros(max_iter)
	for i in range(max_iter):
		if i < N:
			temp_sum += reward[i]
			throughput[i] = temp_sum / (i+1)
		else:
			temp_sum  += reward[i] - reward[i-N]
			throughput[i] = temp_sum / N
	return throughput

my_agent_throughputs = {}
agent_rewards = []
agent_actions = []
threes_buffers = []
twos_buffers = []
ones_buffers = []
agent_collisions = []
agent_throughputs = []
sum_throughputs = {}
numberOfDRLs = 3
N = 1000
for j in range(numberOfDRLs):
    agent_rewards.append(np.loadtxt('rewards/'+str(j)+'_M20.txt'))
    max_iter = len(agent_rewards[0])
    agent_actions.append(np.loadtxt('rewards/action' + str(j) + '_M20.txt'))
    threes_buffers.append(np.loadtxt('rewards/threes_buffer' + str(j) + '_M20.txt'))
    twos_buffers.append(np.loadtxt('rewards/twos_buffer' + str(j) + '_M20.txt'))
    ones_buffers.append(np.loadtxt('rewards/ones_buffer' + str(j) + '_M20.txt'))
    agent_throughputs.append(cal_throughput(max_iter, N, agent_rewards[j]))

x = np.linspace(0, max_iter, max_iter)

    
	
sum_throughputs  = sum(agent_throughputs)

fig, axs = plt.subplots(4, 1, sharex=True)
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0)

# Plot each graph, and manually set the y tick values
#axs[0].set_yticks(np.arange(1, 1.1, 1))
axs[0].set_ylim(0.1, 1.1)
axs[0].set_ylabel('Throughput')
axs[0].grid()
plt.xlabel('Time')
axs[1].set_ylabel('Buffer size of \r\n priority 3 packets')
axs[2].set_ylabel('Buffer size of \r\n priority 2 packets')
axs[3].set_ylabel('Buffer size of \r\n priority 1 packets')
axs[1].grid()
axs[2].grid()
axs[3].grid()

axs[0].plot(sum_throughputs, color='y', lw=1, label='sum')

for j in range(numberOfDRLs):
    col = '#'
    for x in range(j):
        col += '0'
    col += 'a'
    for x in range(5 - j):
        col += '0'
    axs[0].plot(agent_throughputs[j], color=col, lw=1, label='agent'+str(j))
    #axs[0].plot(agent_actions[j]*(.5 + j/2 + agent_collisions[j]/5) + 1 + j/10, '.', color=col, lw=.1)
handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles, labels, loc='lower left', ncol=3, bbox_to_anchor=(0,1))

axs[0].set_xlim((0, max_iter))
for j in range(numberOfDRLs):
    col = '#'
    for x in range(j):
        col += '0'
    col += 'a'
    for x in range(5 - j):
        col += '0'
    axs[1].plot(threes_buffers[j], color=col, lw=1, label='agent'+str(j))
for j in range(numberOfDRLs):
    col = '#'
    for x in range(j):
        col += '0'
    col += 'a'
    for x in range(5 - j):
        col += '0'
    axs[2].plot(twos_buffers[j], color=col, lw=1, label='agent'+str(j))
for j in range(numberOfDRLs):
    col = '#'
    for x in range(j):
        col += '0'
    col += 'a'
    for x in range(5 - j):
        col += '0'
    axs[3].plot(ones_buffers[j], color=col, lw=1, label='agent'+str(j))
plt.show()




















