import random as rd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint




# rd.seed(12)








class Particle():
    def __init__(self):
        x = (-1) ** bool(rd.getrandbits(1)) * rd.random() * 1000
        y = (-1) ** bool(rd.getrandbits(1)) * rd.random() * 1000
        self.position = np.array([x, y])
        self.pBest_position = self.position
        self.pBest_value = float('inf')
        self.velocity = np.array([0,0])

        self.nBest_value = float('inf')
        self.nBest_position = np.array([rd.random() * 50, rd.random() * 50])

    def update(self):
        self.position = self.position + self.velocity









class Space():
    def __init__(self, target, target_error, n_particles, fitness):
        self.target = target
        self.target_error = target_error
        self.n_particles = n_particles
        self.fitness = fitness
        self.particles = []
        self.topology = []

        self.gBest_value = float('inf')
        self.gBest_position = np.array([rd.random() * 50, rd.random() * 50])

    def set_pBest(self):
        for particle in self.particles:
            fitness_candidate = self.fitness(particle)
            if particle.pBest_value > fitness_candidate:
                particle.pBest_value = fitness_candidate
                particle.pBest_position = particle.position

    def set_gBest(self):
        for particle in self.particles:
            best_fitness_candidate = self.fitness(particle)
            if self.gBest_value > best_fitness_candidate:
                self.gBest_value = best_fitness_candidate
                self.gBest_position = particle.position

    def set_nBest(self):
        for particleId, particle in enumerate(self.particles):

            particle.nBest_value = particle.pBest_value
            # bestNeighbor = particle

            neighborIds = self.topology[particleId]

            for nId in neighborIds:
                if self.particles[nId].pBest_value < particle.pBest_value:
                    particle.nBest_value = self.particles[nId].pBest_value
                    particle.nBest_position = self.particles[nId].pBest_position

            # best_fitness_candidate = self.fitness(particle)
            # bestNeighbor = self.getBestNeighbor(particleId)
            #
            # if bestNeighbor.pBest_value < best_fitness_candidate:
            #     particle.nBest_value = best_fitness_candidate
            #     particle.nBest_position = particle.position

    def update_particles(self):
        for particle in self.particles:
            global c0
            inertial = c0 * particle.velocity
            self_confidence = c1 * rd.random() * (particle.pBest_position - particle.position)
            neighbors_confidence = c2 * rd.random() * (particle.nBest_position - particle.position)
            swarm_confidence = c3 * rd.random() * (self.gBest_position - particle.position)
            new_velocity = inertial + self_confidence + neighbors_confidence + swarm_confidence
            particle.velocity = new_velocity
            particle.update()

    def show_particles(self, iteration):
        print('Iterations: ', iteration)
        print('BestPosition in this time:', self.gBest_position)
        print('BestValue in this time:', self.gBest_value)
        print()

        # for particle in self.particles:
        #     plt.plot(particle.position[0], particle.position[1], 'ro')
        # plt.plot(self.gBest_position[0], self.gBest_position[1], 'bo')
        # plt.show()






def fitness(particle):
    # x = particle.position[0]
    # y = particle.position[1]
    # f =  x**2 + y**2 + 1
    # return f

    return sum( 100.0*(particle.position[i+1]-particle.position[i]**2)**2 + (1-particle.position[i])**2 for i in range(0,len(particle.position)-1) )



def topologyA(particles):
    allNodes = [i for i in range(len(particles))]

    topology = [[] for _ in range(len(particles))]

    for i in range(round(len(particles) / 3)):
        neighbors = [i for i in range(i * 3, (i * 3) + 3)]
        nonNeighbors = np.setdiff1d(allNodes, neighbors).tolist()

        topology[i*3].append(neighbors[1])
        topology[i*3].append(neighbors[2])
        topology[i*3+1].append(neighbors[0])
        topology[i*3+1].append(neighbors[2])
        topology[i*3+2].append(neighbors[0])
        topology[i*3+2].append(neighbors[1])

        random1 = rd.randrange(0, 3)


        while True:
            random2 = rd.randrange(0, len(particles) - 3)

            if random1 != random2 and topology[i*3+random1] not in topology[random1]:

                topology[i*3+random1].append(nonNeighbors[random2])
                topology[nonNeighbors[random2]].append(i*3+random1)
                break


    return topology


def topologyB(particles):
    allNodes = [i for i in range(len(particles))]

    topology = [[] for _ in range(len(particles))]

    for i in range(round(len(particles) / 4)):
        neighbors = [i for i in range(i * 4, (i * 4) + 4)]
        nonNeighbors = np.setdiff1d(allNodes, neighbors).tolist()

        topology[i*4].append(neighbors[1])
        topology[i*4].append(neighbors[2])
        topology[i*4].append(neighbors[3])

        topology[i*4+1].append(neighbors[0])
        topology[i*4+1].append(neighbors[2])
        topology[i*4+1].append(neighbors[3])

        topology[i*4+2].append(neighbors[0])
        topology[i*4+2].append(neighbors[1])
        topology[i*4+2].append(neighbors[3])

        topology[i*4+3].append(neighbors[0])
        topology[i*4+3].append(neighbors[1])
        topology[i*4+3].append(neighbors[2])

        random1 = rd.randrange(0, 4)


        while True:
            random2 = rd.randrange(0, len(particles) - 4)

            if random1 != random2 and topology[i*4+random1] not in topology[random1]:

                topology[i*4+random1].append(nonNeighbors[random2])
                topology[nonNeighbors[random2]].append(i*4+random1)
                break


    return topology


def topologyC(particles):
    particleCount = len(particles)
    edgeCount = round((particleCount / 3 * 3) + (particleCount / 3))
    print(edgeCount)

    topology = [[] for _ in range(len(particles))]

    for i in range(round(particleCount)):
        while True:
            random = rd.randrange(0, particleCount)
            if i != random and random not in topology[i]:
                topology[i].append(random)
                topology[random].append(i)
                break

    return topology


def topologyD(particles):
    particleCount = len(particles)
    edgeCount = round((particleCount / 4 * 6) + (particleCount / 4))
    print(edgeCount)

    topology = [[] for _ in range(len(particles))]

    for i in range(round(particleCount)):
        while True:
            random = rd.randrange(0, particleCount)
            if i != random and random not in topology[i]:
                topology[i].append(random)
                topology[random].append(i)
                break

    return topology









totalRuns = 200
c0 = 0.7 # inertial
c1 = 1.0 # particle
c2 = 2.0 # neighbors
c3 = 0.0 # global

n_iterations = 2000
n_particles = 24
target = 0
target_error = 1e-32
topology = topologyA







allIterations = []
allErrors = []
failsCount = 0

for run in range(totalRuns):
    print('Run: ', run + 1)

    space = Space(target, target_error, n_particles, fitness)
    space.particles = [Particle() for _ in range(space.n_particles)]
    space.topology = topology(space.particles)


    iteration = 0
    while iteration < n_iterations:
        space.set_pBest()
        space.set_gBest()
        space.set_nBest()

        # space.show_particles(iteration)

        if abs(space.gBest_value - space.target) <= space.target_error:
            break

        space.update_particles()
        iteration += 1
        if iteration >= n_iterations:
            failsCount += 1


    print("The best solution is: ", space.gBest_position, " in ", iteration, " iterations")
    bestParticle = Particle()
    bestParticle.position = space.gBest_position
    print("Fitness function error: ", '{:.20f}'.format(space.target_error))
    print()

    allIterations.append(iteration)
    allErrors.append(space.target_error)



print('Testing algorithm finished.')
print('Iterations Mean: ', np.mean(allIterations))
print('Errors Mean: ', np.mean(allErrors))
print('Fails Count: ', failsCount)
