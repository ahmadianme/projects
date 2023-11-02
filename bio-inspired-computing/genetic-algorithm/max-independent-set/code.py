import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math


class GeneticAlgorithm:

    def __init__(self, input):
        self.input = np.array(input[1:])

        self.solutionCount = 1000
        self.fitnessGoal = input[:1][0][0]

        self.solutionVariableCount = input[:1][0][0]
        self.crossoverProbability = .05
        self.mutationProbability = .05

        self.selectionStrategy = 'ranking' # top10Percent | rolletWheel | ranking


    def initializeSolutions(self):
        return np.random.randint(0, len(self.input), (self.solutionCount, self.solutionVariableCount))

    def fitness(self, solution):
        solution = np.unique(solution)

        independentCount = 0

        for i in range(len(solution)):
            isIndependent = True
            for j in range(len(solution)):
                if i == j:
                    continue

                if self.input[solution[i]][solution[j]] == 1:
                    isIndependent = False
                    continue

            if isIndependent:
                independentCount += 1

        return independentCount

    def calcFitness(self, solutions):
        fitnessArray = np.apply_along_axis(self.fitness, 1, solutions)

        return fitnessArray

    def sortSolutions(self, fitnessArray, solutions):
        solutions = np.reshape(solutions, (len(solutions), self.solutionVariableCount))
        indexes = np.argsort(fitnessArray)
        indexes = np.flip(indexes)

        return fitnessArray[indexes], solutions[indexes]

    def selectTop10Percent(self, bestSolutions, bestFitness):
        selectSolutionsCount = math.floor(self.solutionCount*.1)
        selectedSolutions = bestSolutions[0:math.floor(self.solutionCount*.1)]

        bestFitnessTop10Percent = bestFitness[0:selectSolutionsCount]
        bestFitnessTop10PercentNormalized = bestFitnessTop10Percent/bestFitnessTop10Percent.sum(keepdims=1)
        bestFitnessTop10PercentNormalized = bestFitnessTop10PercentNormalized * (self.solutionCount - selectSolutionsCount)
        bestFitnessTop10PercentNormalized = np.ceil(bestFitnessTop10PercentNormalized)

        for i in range(len(bestFitnessTop10PercentNormalized)):
            selectedSolutions = np.concatenate((selectedSolutions, np.repeat([selectedSolutions[i]], [bestFitnessTop10PercentNormalized[i]], axis=0)), axis=0)

        if (len(selectedSolutions) > self.solutionCount):
            for i in range(len(selectedSolutions) - self.solutionCount):
                selectedSolutions = np.delete(selectedSolutions, random.randint(0, len(selectedSolutions) - 1), axis=0)

        return selectedSolutions

    def selectRolletWheel(self, bestSolutions, bestFitness):
        bestFitnessNormalized = bestFitness/bestFitness.sum(keepdims=1)
        bestFitnessSummed = [bestFitnessNormalized[0]]

        for i in range(1, len(bestFitnessNormalized)):
            bestFitnessSummed.append(bestFitnessNormalized[i] + bestFitnessSummed[i-1])

        bestFitnessSummed = np.array(bestFitnessSummed)

        selectedSolutions = np.zeros((self.solutionCount, self.solutionVariableCount))

        for i in range(self.solutionCount):
            rNumber = random.uniform(0, 1)
            selectedIndex = np.where(bestFitnessSummed == bestFitnessSummed[bestFitnessSummed >= rNumber][0])[0][0]
            selectedSolutions[i] = bestSolutions[selectedIndex]

        return selectedSolutions.astype('int64')



    def selectRanking(self, bestSolutions, bestFitness):
        selectedSolutions = np.zeros((self.solutionCount, self.solutionVariableCount))

        proportion = .3
        probabilities = np.zeros(self.solutionCount)
        probabilities[0] = .15

        for i in range(1, self.solutionCount):
            probabilities[i] = probabilities[i-1] * .9

        bestFitnessNormalized = probabilities/probabilities.sum(keepdims=1)
        bestFitnessSummed = [bestFitnessNormalized[0]]

        for i in range(1, len(bestFitnessNormalized)):
            bestFitnessSummed.append(bestFitnessNormalized[i] + bestFitnessSummed[i-1])

        bestFitnessSummed = np.array(bestFitnessSummed)

        selectedSolutions = np.zeros((self.solutionCount, self.solutionVariableCount))

        for i in range(self.solutionCount):
            rNumber = random.uniform(0, 1)
            selectedIndex = np.where(bestFitnessSummed == bestFitnessSummed[bestFitnessSummed >= rNumber][0])[0][0]
            selectedSolutions[i] = bestSolutions[selectedIndex]

        return selectedSolutions.astype('int64')

    def crossover(self, solutions):
        for i in range(len(solutions)):
            rNumber = random.uniform(0, 1)

            if rNumber >= 1 - self.crossoverProbability:
                spouse = random.randint(0, len(solutions) - 1)
                breakPoint = random.randint(1, len(solutions[0]) - 1)

                solutions[i][0:breakPoint], solutions[spouse][0:breakPoint] = np.copy(solutions[spouse][0:breakPoint]), np.copy(solutions[i][0:breakPoint])

        return solutions


    def mutate(self, solutions):
        for i in range(len(solutions)):
            rNumber = random.uniform(0, 1)

            if rNumber >= 1 - self.mutationProbability:
                indexes = np.random.randint(0, self.solutionVariableCount, (1, np.random.randint(0, self.solutionVariableCount)))[0]

                for index in indexes:
                    solutions[i][index] = np.random.randint(0, len(self.input))

        return solutions

    def calcDiversity(self, solutions):
        diversity = 0

        if len(solutions) == 0:
            return diversity

        for i in range(len(solutions)):
            for j in range(len(solutions)):
                if i == j:
                    continue

                diversity = diversity + np.sum(np.abs(np.subtract(solutions[i], solutions[j])))

        return diversity

    def plot(self, fitnessMean, fitnessMax, diversity=None):
        plt.figure()

        if diversity is not None and len(diversity):
            plt.subplot(211)

        plt.plot(fitnessMean, label='Fitness Mean')
        plt.plot(fitnessMax, label='Fitness Max')
        plt.title('Fitness / Diversity Plot')
        plt.xlabel('Generation Number')
        plt.ylabel('Fitness')
        plt.legend()

        if diversity is not None and len(diversity):
            plt.subplot(212)
            plt.plot(diversity, label='Diversity')
            plt.xlabel('Generation Number')
            plt.ylabel('Diversity')
            plt.legend()

        plt.show()









with open('input.txt', 'r') as file:
    input = [[int(num) for num in line.split(' ')] for line in file]





generationId = 0
minGenerationIteration = 8

while generationId < minGenerationIteration:
    startTime = time.time()

    genetic = GeneticAlgorithm(input)


    allFitnessMax = []
    allFitnessMean = []
    allDiversities = []


    currentSolutions = genetic.initializeSolutions()

    for generationId in range(100000):
        generationStartTime = time.time()

        fitness = genetic.calcFitness(currentSolutions)

        allFitnessMean.append(np.mean(fitness))
        allDiversities.append(genetic.calcDiversity(currentSolutions))

        bestFitness, bestSolutions = genetic.sortSolutions(fitness, currentSolutions)

        allFitnessMax.append(bestFitness[0])

        print(f"Generation {generationId} - Best Fitness: {int(bestFitness[0])} - Solution: {bestSolutions[0] + 1}")

        if bestFitness[0] >= genetic.fitnessGoal:
            break

        if genetic.selectionStrategy == 'top10Percent':
            currentSolutions = genetic.selectTop10Percent(bestSolutions, bestFitness)
        else:
            if genetic.selectionStrategy == 'rolletWheel':
                currentSolutions = genetic.selectRolletWheel(bestSolutions, bestFitness)
            else:
                if genetic.selectionStrategy == 'ranking':
                    currentSolutions = genetic.selectRanking(bestSolutions, bestFitness)
                else:
                    print('Error: No Selection Strategy specified.')
                    exit()

        currentSolutions = genetic.crossover(currentSolutions)
        currentSolutions = genetic.mutate(currentSolutions)




    print()
    print(f"Found Best:\n- Solution: {bestSolutions[0]}\n- Fitness: {bestFitness[0] + 1}")
    print('Total Time: ' + str(time.time() - startTime))
    print('Generation Time Mean: ' + str((time.time() - startTime) / (generationId+1)))

    if len(allDiversities):
        print(f"Diversity: Min: {np.min(allDiversities)} - Max: {np.max(allDiversities)} - Mean: {np.mean(allDiversities)}")

    print("\n\n\n\n\n\n")


genetic.plot(allFitnessMean, allFitnessMax, allDiversities)
