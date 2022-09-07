import matplotlib.pyplot as plt

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
for a, o in zip(agent_thr, opt_thr):
    temp_a += a
    temp_o += o
    c += 1
    agent.append(temp_a/c)
    opt.append(temp_o/c)

fig, axs = plt.subplots(1, 1, sharex=True)
axs.set_ylabel('Throughput', fontsize=26)
plt.xlabel('Beta (1/Lambda)', fontsize=26)
axs.grid()
for tick in axs.yaxis.get_major_ticks():
    tick.label.set_fontsize(24)
for tick in axs.yaxis.get_major_ticks():
    tick.label.set_fontsize(24)
for tick in axs.xaxis.get_major_ticks():
    tick.label.set_fontsize(24)

print(x)
plt.plot(agent_thr, lw=4, label='agent')
plt.plot(opt_thr, lw=4, label='optimal')
#plt.plot([4, 3, 2, 6], [1, 4, 9, 16])
plt.legend(fontsize=26)
plt.show()