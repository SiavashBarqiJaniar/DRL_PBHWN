from math import *
import numpy as np
#print(np.random.exponential(scale=600, size=None))

class Agent(object):

    def __init__(self, initial_packets, test, NN, beta):
        self.NN = NN
        self.packets = initial_packets
        self.throughput = 0
        self.reward_list = []
        self.action_list = []
        self.no_collisions = 0
        self.buffer_history = [initial_packets]
        self.collisions = []
        self.arrival_time = 0
        self.lambdaa = np.random.randint(1,10)
        self.real_packets = initial_packets
        self.test = test
        self.over_100 = False
        self.over = 100
        self.pckt_ratio = [1 for j in range(NN-1)]
        self.opp_buffer = [initial_packets for j in range(NN-1)]
        self.no_continuous_colls = 0
        self.beta = beta
        self.arrival_time_history = []

    def tx(self, a, cap, t, idd, ep):
        if t == self.arrival_time:
            self.arrival_time_history.append(self.arrival_time)
            #p = str(idd) + ' '
            if t > 0: #and not self.over_100:
                self.real_packets += 40#40#20
                self.packets = self.real_packets
            if self.packets > self.over:
                self.over_100 = True
                self.over += 100
            elif self.packets <= 0:
                self.over_100 = False
            self.arrival_time = t + 1 + min(int(np.random.exponential(scale=self.beta, size=None)),1000+self.beta)
        for j in range(self.NN-1):
            if self.real_packets != 0:
                self.pckt_ratio[j] = self.opp_buffer[j]/self.real_packets
            else:
                self.pckt_ratio[j] = self.opp_buffer[j]/0.001
        if a == 1 and cap == 0:
            self.packets -= 1
            if self.real_packets > 0:
                self.real_packets -= 1
            else:
                self.real_packets = 0
            
        self.buffer_history.append(self.packets)

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
        return self.opp_buffer