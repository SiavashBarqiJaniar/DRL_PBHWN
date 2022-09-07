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
agent_buffers = []
agent_collisions = []
agent_throughputs = []
sum_throughputs = {}
numberOfDRLs = 5
N = 1000
for j in range(numberOfDRLs):
    agent_rewards.append(np.loadtxt('rewards/agent'+str(j)+'_M20.txt'))
    max_iter = len(agent_rewards[0])
    agent_actions.append(np.loadtxt('rewards/agent_action' + str(j) + '_M20.txt'))
    agent_buffers.append(np.loadtxt('rewards/agent_buffer' + str(j) + '_M20.txt'))
    agent_collisions.append(np.loadtxt('rewards/agent_collision' + str(j) + '_M20.txt'))
    agent_throughputs.append(cal_throughput(max_iter, N, agent_rewards[j]))

x = np.linspace(0, max_iter, max_iter)
    
	
sum_throughputs  = sum(agent_throughputs)
avg_thr = np.ones(max_iter)
avg_thr *= sum(sum_throughputs)/max_iter

fig, axs = plt.subplots(2, 1, sharex=True)
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0)

# Plot each graph, and manually set the y tick values
#axs[0].set_yticks(np.arange(1, 1.1, 1))
axs[0].set_ylim(0, .7)
axs[0].set_ylabel('Throughput', fontsize=26)
axs[1].set_ylabel('Buffer size', fontsize=26)
plt.xlabel('Time', fontsize=26)
axs[0].grid()
for tick in axs[0].yaxis.get_major_ticks():
    tick.label.set_fontsize(24)
for tick in axs[1].yaxis.get_major_ticks():
    tick.label.set_fontsize(24)
for tick in axs[1].xaxis.get_major_ticks():
    tick.label.set_fontsize(24)

axs[1].grid()

axs[0].plot(sum_throughputs, color='y', lw=1, label='sum')
axs[0].plot(avg_thr, color='goldenrod', lw=1.5, label='averaged total throughput')
print(sum_throughputs)
print(avg_thr)

for j in range(numberOfDRLs):
    col = '#'
    for x in range(j):
        col += '0'
    col += 'a'
    for x in range(5 - j):
        col += '0'
    axs[0].plot(agent_throughputs[j], color=col, lw=1, label='user '+str(j))
    #axs[0].plot(agent_actions[j]*(.5 + j/2 + agent_collisions[j]/5) + 1 + j/10, '.', color=col, lw=.1)
handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles, labels, loc='lower left', ncol=4, bbox_to_anchor=(0,1), fontsize = 20)

axs[0].set_xlim((0, max_iter))
#x = np.linspace(0,40000,len(agent_buffers))
#agent_buffers = np.array(agent_buffers)
#print(agent_buffers.shape)
#print(x.shape)
for j in range(numberOfDRLs):
    col = '#'
    for x in range(j):
        col += '0'
    col += 'a'
    for x in range(5 - j):
        col += '0'
    #axs[1].stem(agent_buffers[j], linefmt='C'+str(j), markerfmt=' ', use_line_collection=True)
    axs[1].plot(agent_buffers[j], color=col, lw=1, label='user '+str(j))
plt.show()




















