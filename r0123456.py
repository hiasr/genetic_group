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

        # Your code here.

        population = self.init_population()
        iters = 0
        while iters < self.numIters:
            meanObjective = 0.0
            bestObjective = 0.0
            bestSolution = np.array([1,2,3,4,5])

            # Your code here.

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
        return 0

    """Initialize population as random permutations"""
    def init_population(self):
        population = np.zeros((self.lambdaa, self.lambdaa))
        for i in range(self.lambdaa):
            population[i] = self.rng.permutation(self.lambdaa)
        return population

    """ Perform k-tournament selection to select pairs of parents. """
    def selection(self, population):
        selected = np.zeros((self.mu, 2))
        for ii in range( self.mu ):
            ri = random.choices(range(np.size(population,0)), k = self.k)
            min = np.argmin( self.objf(population[ri, :]) )
            selected[ii,:] = population[ri[min],:]
        return selected

    """ Perform box crossover as in the slides. """
    def crossover(self, selected):
        weights = 3*np.random.rand(self.lambdaa,2) - 1
        offspring = np.zeros((self.lambdaa, 2))
        lc = lambda x, y, w: np.clip(x + w * (y-x), 0, self.intMax)
        for ii, _ in enumerate(offspring):
            offspring[ii,:] = lc(selected[2*ii, :], selected[2*ii+1, :], weights[ii, :])
        return offspring

    """ Perform mutation, adding a random Gaussian perturbation. """
    def mutation(self, offspring, alpha):
        ii = np.where(np.random.rand(np.size(offspring,0)) <= alpha)
        offspring[ii,:] = offspring[ii,:] + 10*np.random.randn(np.size(ii),2)
        offspring[ii,0] = np.clip(offspring[ii,0], 0, self.intMax)
        offspring[ii,1] = np.clip(offspring[ii,1], 0, self.intMax)
        return offspring

    """ Eliminate the unfit candidate solutions. """
    def elimination(self, joinedPopulation, keep):
        fvals = self.objf(joinedPopulation)
        perm = np.argsort(fvals)
        survivors = joinedPopulation[perm[0:keep-1],:]
        return survivors
    
    def objf(self, permutation):
        result = 0
        for i in range(1, len(permutation)):
            result += self.distanceMatrix[permutation[i-1]][permutation[i]]
        return result

tsp = r0123456()
print(tsp.optimize("datasets/tour50.csv"))
