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
    agent_rewards.append(np.loadtxt('rewards/agent'+str(j)+'_M20.txt'))
    max_iter = len(agent_rewards[0])
    agent_actions.append(np.loadtxt('rewards/agent_action' + str(j) + '_M20.txt'))
    threes_buffers.append(np.loadtxt('rewards/agent_threes_buffer' + str(j) + '_M20.txt'))
    twos_buffers.append(np.loadtxt('rewards/agent_twos_buffer' + str(j) + '_M20.txt'))
    ones_buffers.append(np.loadtxt('rewards/agent_ones_buffer' + str(j) + '_M20.txt'))
    agent_collisions.append(np.loadtxt('rewards/agent_collision' + str(j) + '_M20.txt'))
    agent_throughputs.append(cal_throughput(max_iter, N, agent_rewards[j]))

x = np.linspace(0, max_iter, max_iter)
sum_throughputs  = sum(agent_throughputs)

q = .6
agent_opt = np.ones(max_iter)*(1-.3)*(1-q)
aloha_opt = np.ones(max_iter)*(1-.3)*q
tdma_opt = np.ones(max_iter)*.3*(1-q)
sum_opt = agent_opt + aloha_opt + tdma_opt

agent_avg = np.ones(max_iter)*sum(agent_throughputs[0])/max_iter
aloha_avg = np.ones(max_iter)*sum(agent_throughputs[1])/max_iter
tdma_avg = np.ones(max_iter)*sum(agent_throughputs[2])/max_iter
sum_avg = np.ones(max_iter)*sum(sum_throughputs)/max_iter

fig, axs = plt.subplots(4, 1, sharex=True)
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0)

# Plot each graph, and manually set the y tick values
#axs[0].set_yticks(np.arange(1, 1.1, 1))
axs[0].set_ylim(0, .6)
axs[0].set_ylabel('Throughput', fontsize=20)
axs[0].grid()
plt.xlabel('Time', fontsize=20)
axs[1].set_ylabel('High', fontsize=20)
axs[2].set_ylabel('Medium', fontsize=20)
axs[3].set_ylabel('Low', fontsize=20)
for tick in axs[0].yaxis.get_major_ticks():
    tick.label.set_fontsize(20)
for tick in axs[1].yaxis.get_major_ticks():
    tick.label.set_fontsize(20)
for tick in axs[2].yaxis.get_major_ticks():
    tick.label.set_fontsize(20)
for tick in axs[3].yaxis.get_major_ticks():
    tick.label.set_fontsize(20)
for tick in axs[3].xaxis.get_major_ticks():
    tick.label.set_fontsize(20)
axs[1].grid()
axs[2].grid()
axs[3].grid()

axs[0].plot(sum_throughputs, color='y', lw=1, label='sum')

axs[0].plot(agent_throughputs[0], color='#a00000', lw=1, label='intelligent')
axs[0].plot(agent_throughputs[1], color='#0a0000', lw=1, label='slotted-ALOHA')
axs[0].plot(agent_throughputs[2], color='#00a000', lw=1, label='TDMA')

axs[0].plot(agent_avg, color='r', lw=2, ls='--', label='intell-avg')
axs[0].plot(aloha_avg, color='black', lw=2, ls='--', label='ALOHA-avg')
axs[0].plot(tdma_avg, color='lawngreen', lw=2, ls='--', label='TDMA-avg')
axs[0].plot(sum_avg, color='goldenrod', lw=2, ls='--', label='sum-avg')

handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles, labels, loc='lower left', ncol=4, bbox_to_anchor=(0,1), fontsize=20)

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




















