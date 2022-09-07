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
rewards = []
actions = []
buffers = []
throughputs = []
agent_sum_throughputs = {}
numberOfDRLs = 3
N = 1000
for j in range(numberOfDRLs):
    agent_rewards.append(np.loadtxt('rewards/agent'+str(j)+'_M20.txt'))
    max_iter = len(agent_rewards[0])
    agent_actions.append(np.loadtxt('rewards/agent_action' + str(j) + '_M20.txt'))
    agent_buffers.append(np.loadtxt('rewards/agent_buffer' + str(j) + '_M20.txt'))
    agent_collisions.append(np.loadtxt('rewards/agent_collision' + str(j) + '_M20.txt'))
    agent_throughputs.append(cal_throughput(max_iter, N, agent_rewards[j]))
    rewards.append(np.loadtxt('rewards/'+str(j)+'_M20.txt'))
    actions.append(np.loadtxt('rewards/action' + str(j) + '_M20.txt'))
    buffers.append(np.loadtxt('rewards/buffer' + str(j) + '_M20.txt'))
    throughputs.append(cal_throughput(max_iter, N, rewards[j]))

x = np.linspace(0, max_iter, max_iter)

agent_sum_throughputs  = sum(agent_throughputs)
sum_throughputs  = sum(throughputs)
a_sum = 0
agent_sum = []
for x in agent_sum_throughputs:
    a_sum += x
    agent_sum.append( a_sum/(len(agent_sum) + 1) )
summm = 0
summ = []
for x in sum_throughputs:
    summm += x
    summ.append(summm/( (len(summ) + 1) ))

fig, axs = plt.subplots(1, 1, sharex=True)
axs.set_ylim(0, 1.1)
axs.set_ylabel('Throughput', fontsize=26)
plt.xlabel('Time', fontsize=26)
axs.grid()
for tick in axs.yaxis.get_major_ticks():
    tick.label.set_fontsize(24)
for tick in axs.yaxis.get_major_ticks():
    tick.label.set_fontsize(24)
for tick in axs.xaxis.get_major_ticks():
    tick.label.set_fontsize(24)

axs.plot(agent_sum, lw=4, label='proposed')
axs.plot(summ, lw=4, label='genie-aided')

handles, labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower left', ncol=4, bbox_to_anchor=(0,1), fontsize=26)

axs.set_xlim((0, max_iter))
plt.show()




















