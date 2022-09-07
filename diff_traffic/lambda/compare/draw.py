import matplotlib.pyplot as plt
import numpy as np

agent_thr = []
opt_thr = []
with open('agent_throughput.txt', 'r') as f:
    for lines in f:
        item = lines[:-1]
        agent_thr.append(float(item)*100)
with open('opt_throughput.txt', 'r') as f:
    for lines in f:
        item = lines[:-1]
        opt_thr.append(float(item)*100)

temp_a = 0
temp_o = 0
c = 0
x = []
agent = []
opt = []

for i in range(len(agent_thr)):
    if i < 2:
        agent.append(np.mean(agent_thr[:i+1]))
        opt.append(np.mean(opt_thr[:i+1]))
    else:
        agent.append(np.mean(agent_thr[i-2:i+1])) # mean over (i-3), (i-2), (i-1), i
        opt.append(np.mean(opt_thr[i-2:i+1]))

for i in range(len(agent_thr)):
    x.append(401 - i*8)

fig, axs = plt.subplots(1, 1, sharex=True)
axs.set_ylabel('Throughput percentage', fontsize=26)
plt.xlabel('Beta (1/Lambda)', fontsize=26)
axs.set_xlim(420, -10)
axs.grid()
for tick in axs.yaxis.get_major_ticks():
    tick.label.set_fontsize(24)
for tick in axs.yaxis.get_major_ticks():
    tick.label.set_fontsize(24)
for tick in axs.xaxis.get_major_ticks():
    tick.label.set_fontsize(24)

if len(x) != len(opt):
    print("WWWWWOOOOOOOOOOOOOOOOOWWWWWWWWWWWWWWWWWWWWW!!!!!!!!!!!!")

print(x)
plt.plot(x, agent, lw=4, label='proposed')
plt.plot(x, opt, lw=4, label='genie-aided')
#plt.plot([4, 3, 2, 6], [1, 4, 9, 16])
plt.legend(fontsize=26)
plt.show()