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
aloha_rewards = []
agent_throughputs = []
aloha_throughputs = []
numberOfDRLs = 1
N = 1000
agent_rewards.append(np.loadtxt('rewards/agent0_M20.txt'))
aloha_rewards.append(np.loadtxt('rewards/aloha_occ0_M20.txt'))
max_iter = len(agent_rewards[0])
agent_throughputs.append(cal_throughput(max_iter, N, agent_rewards[0]))
aloha_throughputs.append(cal_throughput(max_iter, N, aloha_rewards[0]))

x = np.linspace(0, max_iter, max_iter)

avg_agent = sum(agent_throughputs[0])/max_iter*np.ones(max_iter)
avg_aloha = sum(aloha_throughputs[0])/max_iter*np.ones(max_iter)
print(avg_aloha)

fig, axs = plt.subplots(1, 1, sharex=True)
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0)

axs.set_ylim(0, 25)
axs.set_ylabel('Channel occupancy (%)', fontsize=26)
plt.xlabel('Time', fontsize=26)
axs.grid()
for tick in axs.yaxis.get_major_ticks():
    tick.label.set_fontsize(24)
for tick in axs.xaxis.get_major_ticks():
    tick.label.set_fontsize(24)

axs.plot(agent_throughputs[0]*100, color='r', lw=1, label='intelligent')
axs.plot(aloha_throughputs[0]*100, color='b', lw=1, label='slotted-aloha')

axs.plot(avg_agent*100, color='darkred', lw=2.5, label='intell-avg')
axs.plot(avg_aloha*100, color='darkblue', lw=2.5, label='aloha-avg')

handles, labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower left', ncol=2, bbox_to_anchor=(0,1), fontsize=24)
axs.set_xlim((0, max_iter))
plt.show()




















