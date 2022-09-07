from math import *
import numpy as np
#print(np.random.exponential(scale=600, size=None))

class Agent(object):

    def __init__(self, NN, beta, threes, twos, ones):
        initial_packets = threes*3 + 2*twos + ones
        self.NN = NN
        self.threes = threes
        self.twos = twos
        self.ones = ones
        self.packets = initial_packets
        self.throughput = 0
        self.reward_list = []
        self.action_list = []
        self.ones_history = [ones]
        self.twos_history = [twos]
        self.threes_history = [threes]
        self.arrival_time_threes = 0
        self.arrival_time_twos = 0
        self.arrival_time_ones = 0
        self.lambdaa = np.random.randint(1,10)
        self.real_packets = initial_packets
        self.over_100 = False
        self.over = 100
        self.three_ratio = [1 for j in range(NN-1)]
        self.two_ratio = [1 for j in range(NN-1)]
        self.one_ratio = [1 for j in range(NN-1)]
        self.opps_buffers = [[ones for j in range(NN-1)], [twos for j in range(NN-1)], [threes for j in range(NN-1)]]
        self.no_continuous_colls = 0
        self.beta = beta

    def tx(self, a, cap, t, idd):
        if t == self.arrival_time_threes:
            if t > 0:
                self.threes += 5
            if self.packets > self.over:
                self.over_100 = True
                self.over += 100
            elif self.packets <= 0:
                self.over_100 = False
            beta = self.beta*3
            self.arrival_time_threes = t + 1 + min(int(np.random.exponential(scale=beta, size=None)),1000+beta)
            self.real_packets = 3*self.threes + 2*self.twos + self.ones
            self.packets = self.real_packets
        if t == self.arrival_time_twos:
            if t > 0:
                self.twos += 10
            if self.packets > self.over:
                self.over_100 = True
                self.over += 100
            elif self.packets <= 0:
                self.over_100 = False
            beta = self.beta*2
            self.arrival_time_twos = t + 1 + min(int(np.random.exponential(scale=beta, size=None)),1000+beta)
            self.real_packets = 3*self.threes + 2*self.twos + self.ones
            self.packets = self.real_packets
        if t == self.arrival_time_ones:
            if t > 0:
                self.ones += 20
            if self.packets > self.over:
                self.over_100 = True
                self.over += 100
            elif self.packets <= 0:
                self.over_100 = False
            self.arrival_time_ones = t + 1 + min(int(np.random.exponential(scale=self.beta, size=None)),1000+self.beta)
            self.real_packets = 3*self.threes + 2*self.twos + self.ones
            self.packets = self.real_packets
        if a == 1 and cap == 0:
            if self.threes > 0:
                self.threes -= 1
                self.packets -= 3
            elif self.twos > 0:
                self.twos -= 1
                self.packets -= 2
            elif self.ones > 0:
                self.ones -= 1
                self.packets -= 1
            else:
                self.packets -= 1
        self.real_packets = 3*self.threes + 2*self.twos + self.ones
        for j in range(self.NN-1):
            if self.threes != 0:
                self.three_ratio[j] = self.opps_buffers[2][j]/self.threes
            else:
                self.three_ratio[j] = self.opps_buffers[2][j]/0.001
            if self.twos != 0:
                self.two_ratio[j] = self.opps_buffers[1][j]/self.twos
            else:
                self.two_ratio[j] = self.opps_buffers[1][j]/0.001
            if self.ones != 0:
                self.one_ratio[j] = self.opps_buffers[0][j]/self.ones
            else:
                self.one_ratio[j] = self.opps_buffers[0][j]/0.001
        self.ones_history.append(self.ones)
        self.twos_history.append(self.twos)
        self.threes_history.append(self.threes)

    def clc_throughput(self):
        N = 1000
        i = len(self.reward_list)
        if i < N:
            r = sum(self.reward_list)
            self.throughput = r/i
        else:
            r = sum(self.reward_list[-1000:]) # from 1000th to the last item through the last item
            self.throughput = r/N

    def get_opp_buffer(self):
        return self.opps_buffers