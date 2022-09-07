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
numberOfDRLs = 3
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

q = .4
agent_opt = np.ones(max_iter)*(1-.3)*(1-q)
aloha_opt = np.ones(max_iter)*(1-.3)*q
tdma_opt = np.ones(max_iter)*.3*(1-q)
sum_opt = agent_opt + aloha_opt + tdma_opt

agent_avg = np.ones(max_iter)*sum(agent_throughputs[0])/max_iter
aloha_avg = np.ones(max_iter)*sum(agent_throughputs[1])/max_iter
tdma_avg = np.ones(max_iter)*sum(agent_throughputs[2])/max_iter
sum_avg = np.ones(max_iter)*sum(sum_throughputs)/max_iter
    


fig, axs = plt.subplots(1, 1, sharex=True)
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0)

# Plot each graph, and manually set the y tick values
axs.set_ylim(0, 1.1)
axs.set_xlim(0, max_iter)
axs.set_ylabel('Throughput', fontsize=26)
plt.xlabel('Time', fontsize=26)
axs.grid()
for tick in axs.yaxis.get_major_ticks():
    tick.label.set_fontsize(24)
for tick in axs.xaxis.get_major_ticks():
    tick.label.set_fontsize(24)

axs.plot(sum_throughputs, color='y', lw=1, label='sum')


#axs.plot(agent_avg, color='r', lw=2, label='intell-avg')
#axs.plot(aloha_avg, color='black', lw=2, label='aloha-avg')
#axs.plot(tdma_avg, color='g', lw=2, label='tdma-avg')
#axs.plot(sum_avg, color='y', lw=2, label='sum-avg')

for j in range(numberOfDRLs):
    col = '#'
    for x in range(j):
        col += '0'
    col += 'a'
    for x in range(5 - j):
        col += '0'
    axs.plot(agent_throughputs[j], color=col, lw=1, label='user '+str(j))
    #axs[0].plot(agent_actions[j]*(.5 + j/2 + agent_collisions[j]/5) + 1 + j/10, '.', color=col, lw=.1)

axs.plot(agent_avg, color='r', lw=2, label='intell-avg')
axs.plot(aloha_avg, color='black', lw=2, label='aloha-avg')
axs.plot(tdma_avg, color='lawngreen', lw=2, label='tdma-avg')
axs.plot(sum_avg, color='goldenrod', lw=2, label='sum-avg')

handles, labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower left', ncol=4, bbox_to_anchor=(0,1), fontsize=22)

plt.show()




















