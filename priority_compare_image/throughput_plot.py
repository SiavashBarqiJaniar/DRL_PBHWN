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
    agent_throughputs.append(cal_throughput(max_iter, N, agent_rewards[j]))
    rewards.append(np.loadtxt('rewards/'+str(j)+'_M20.txt'))
    throughputs.append(cal_throughput(max_iter, N, rewards[j]))

x = np.linspace(0, max_iter, max_iter)

agent_sum_throughputs  = sum(agent_throughputs)
sum_throughputs  = sum(throughputs)

fig, axs = plt.subplots(1, 1, sharex=True)
axs.set_ylim(0, 1.1)
axs.set_ylabel('Throughput')
plt.xlabel('Time')
axs.grid()

axs.plot(agent_sum_throughputs, color='y', lw=1, label='agents')
axs.plot(sum_throughputs, color='r', lw=1, label='optimum')

handles, labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower left', ncol=4, bbox_to_anchor=(0,1))

axs.set_xlim((0, max_iter))
plt.show()




















