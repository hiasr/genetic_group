import Reporter
import random
import numpy as np
#
# Modify the class name to match your student number.
class r0123456:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.alpha = 0.05           # Mutation probability
        self.lambdaa = 100          # Population size
        self.mu = self.lambdaa * 2  # Offspring size
        self.k = 5                  # Tournament selection
        self.intMax = 500           # Boundary of the domain, not intended to be changed.
        self.numIters = 10          # Maximum number of iterations
        self.rng = np.random.default_rng(72211889822662786358204883137771642604)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.       
        file = open(filename)
        self.distanceMatrix = np.loadtxt(file, delimiter=",")
        self.n = len(self.distanceMatrix)
        file.close()

        # population size: (50,100)
        population = self.init_population()
        iters = 0
        while iters < self.numIters:
            fvals = self.objf(population)
            meanObjective = fvals.mean()
            bestObjectiveIndex = fvals.argmax()
            bestObjective = fvals[bestObjectiveIndex]
            bestSolution = population[bestObjectiveIndex,:]

            # Your code here.
            selected = self.selection(population, self.k)
            #print(selected)
            offspring = self.crossover(selected)
            #print(offspring)
            joinedPopulation = np.vstack((self.mutation(offspring, self.alpha), population))
            #print(joinedPopulation)
            population = self.elimination(joinedPopulation, self.lambdaa)
            #print(population)

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution 
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break
            iters += 1
        # Your code here.
        return max(self.objf(population))

    """Initialize population as random permutations"""
    def init_population(self):
        population = np.zeros((self.lambdaa, self.n), dtype=np.int64)
        for i in range(self.lambdaa):
            population[i,:] = self.rng.permutation(self.n)
        return population

    """ Perform k-tournament selection to select pairs of parents. """
    def selection(self, population, k):
        selected = np.zeros((self.mu, self.n))
        for ii in range( self.mu ):
            ri = self.rng.choice(range(np.size(population,0)), size=(k,))
            min = np.argmin( self.objf(population[ri, :]) )
            selected[ii,:] = population[ri[min],:]
        return selected

    def crossover(self, selected):
        offspring = np.zeros((self.mu, self.n), dtype=np.int64)
        
        for n in range(self.mu//2):
            start, end = sorted(self.rng.integers(self.n,size=(2,)))
            p1, p2 = 2*n, 2*n+1
            for _ in range(2):
                offspring[p1,start:end] = selected[p1,start:end]
                i = 0
                for j_ in range(self.n):
                    j = (end+j_)%self.n
                    if not selected[p2,j] in offspring[p1,:]:
                        offspring[p1,(end+i)%self.n] = selected[p2,j]
                        i += 1
                p1,p2 = p2,p1
        return offspring

    """ Perform mutation, adding a random Gaussian perturbation. """
    def mutation(self, offspring, alpha):
        ii = np.where(self.rng.random(np.size(offspring,0)) <= alpha)[0]
        for i in ii:
            a, b = sorted(self.rng.integers(self.n,size=(2,)))
            offspring[i, a:b+1] = np.flip(offspring[i, a:b+1])
        return offspring

    """ Eliminate the unfit candidate solutions. """
    def elimination(self, joinedPopulation, keep):
        fvals = self.objf(joinedPopulation)
        perm = np.argsort(fvals)
        survivors = joinedPopulation[perm[0:keep-1],:]
        return survivors
    
    def objf(self, permutation):
        m = np.size(permutation,0)
        fvals = np.zeros(m)
        for k in range(m):
            for j in range(1, self.n):
                 fvals[k] += self.distanceMatrix[permutation[k,j-1]][permutation[k,j]]
        return fvals



tsp = r0123456()
print(tsp.optimize("datasets/tour50.csv"))
