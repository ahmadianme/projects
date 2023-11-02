import graph
import networkx as nx
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt





m = 2
iterations = 10
minDegree = 2
gammas = [
    2,
    2.5,
    3,
    5
]

Ns = [
    int(1e1),
    int(1e2),
    int(1e3),
    int(1e4),
    int(1e5),
]

NsLabel = []
NsLabel = ['{:.0e}'.format(float(x)) for x in Ns]





avgDistances = {}
allAvgDistances = {}

for gamma in gammas:
    avgDistances[gamma] = []
    allAvgDistances[gamma] = []




for N in Ns:
    for gamma in gammas:
        avgDistancesSum = 0


        for i in range(iterations):
            print('N: ' + str(N) + ' gamma: ' + str(gamma))


            gammasSeq = []
            while len(gammasSeq) < N:
                nextval = int(nx.utils.powerlaw_sequence(1, gamma)[0]) + 1 #100 nodes, power-law exponent 2.5
                if nextval >= minDegree and nextval < N - 1:
                    gammasSeq.append(nextval)


            gammasSeq = (gammasSeq-np.min(gammasSeq))/(np.max(gammasSeq)-np.min(gammasSeq))
            gammasSeq = new_items = [0.00000001 if x == 0 else x for x in gammasSeq]


            bbmodel=graph.create_bb_model(m, N, gammasSeq)
            # G=nx.from_numpy_matrix(np.mat(bbmodel.mat()))
            G = nx.from_numpy_matrix(np.mat(bbmodel.mat()), parallel_edges=False, create_using=nx.Graph())


            G = nx.Graph(G) # remove parallel edges if exists
            G.remove_edges_from(nx.selfloop_edges(G))


            avgDistance = nx.average_shortest_path_length(G)
            avgDistancesSum += avgDistance


            degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
            mean_degree = mean(degree_sequence)
            print(mean_degree)


        allAvgDistances[gamma].append(avgDistancesSum / iterations)

print(allAvgDistances)







for gamma in gammas:
    plt.plot([0] + NsLabel, [0] + allAvgDistances[gamma], label="Gamma: " + str(gamma), marker='o', markersize=6)

plt.xlabel('N')
plt.ylabel('<d>')
plt.title('Average Distance on network size of N')
plt.legend()
plt.show()
