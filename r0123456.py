import Reporter
import random
import numpy as np

# Modify the class name to match your student number.
class r0123456:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.alpha = 0.05           # Mutation probability
        self.lambdaa = 100          # Population size
        self.mu = self.lambdaa * 2  # Offspring size
        self.k = 5                  # Tournament selection
        self.intMax = 500           # Boundary of the domain, not intended to be changed.
        self.numIters = 20          # Maximum number of iterations
        self.rng = np.random.default_rng()

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
            meanObjective = 0.0
            bestObjective = 0.0
            bestSolution = np.array([1,2,3,4,5])

            # Your code here.
            selected = self.selection(population, self.k)
            offspring = self.crossover(selected)
            joinedPopulation = np.vstack((self.mutation(offspring, self.alpha), population))
            population = self.elimination(joinedPopulation, self.lambdaa)

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
            ri = random.choices(range(np.size(population,0)), k = k)
            min = np.argmin( self.objf(population[ri, :]) )
            selected[ii,:] = population[ri[min],:]
        return selected

    """ Perform box crossover as in the slides. """
    def old_crossover(self, selected):
        weights = 3*np.random.rand(self.lambdaa,2) - 1
        offspring = np.zeros((self.lambdaa, 2))
        lc = lambda x, y, w: np.clip(x + w * (y-x), 0, self.intMax)
        for ii, _ in enumerate(offspring):
            offspring[ii,:] = lc(selected[2*ii, :], selected[2*ii+1, :], weights[ii, :])
        return offspring

    def crossover(self, selected):
        i = self.rng.integers(self.n)
        j = self.rng.integers(i,self.n)
        offspring = np.zeros(selected.shape, dtype=np.int64)
        
        for k in range(2):
            offspring[k,i:j] = selected[k,i:j]
            ii = 0
            for jj in range(self.n):
                if not selected[(k+1)%2,(j+jj)%self.n] in offspring[k,:]:
                    offspring[k,(j+ii)%self.n] = selected[(k+1)%2,(j+jj)%self.n]
                    ii += 1
        return offspring

    """ Perform mutation, adding a random Gaussian perturbation. """
    def mutation(self, offspring, alpha):
        ii = np.where(np.random.rand(np.size(offspring,0)) <= alpha)
        a, b = random.choices(range(offspring.size), k=2)
        offspring[a], offspring[b] = offspring[b], offspring[a]
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
            print(permutation[k,:])
            for j in range(1, self.n):
                 fvals[k] += self.distanceMatrix[permutation[k,j-1]][permutation[k,j]]
        return fvals



tsp = r0123456()
print(tsp.optimize("datasets/tour50.csv"))
