import numpy as np
import cv2
import random
from copy import deepcopy
from operator import attrgetter


# Check if the circle is in the boundary of the image frame
def CheckBoundaries(x, y, radius):
    # Case 1: x < 0
    if x < 0:
        if y < 0:
            return (x+radius >= 0) and (y+radius >= 0)
        elif 222 >= y >= 0:
            return x+radius >= 0
        else:
            return (x+radius >= 0) and (222 >= y-radius)

    # Case 2: x is inside the boundary of the Mona Lisa Image
    elif 149 >= x >= 0:
        if y < 0:
            return y+radius >= 0
        elif 222 >= y >= 0:
            return True
        else:
            return 222 >= y-radius

    # Case 3: x is greater than the boundary of the Mona Lisa Image
    else:
        if y < 0:
            return (149 >= x-radius) and (y+radius >= 0)
        elif 222 >= y >= 0:
            return 149 >= x-radius
        else:
            return (149 >= x-radius) and (222 >= y-radius)


# Define the Gene as a class
class Gene:
    def __init__(self, x=0, y=0, radius=1, B=0, G=0, R=0, A=0, randomGene=False):
        self.coordinate = None
        self.radius = None
        self.color = None
        self.A = None
        self.geneArray = self.createGene(x, y, radius, B, G, R, A, randomGene)

    # Create a Gene with the assigned values or randomly
    def createGene(self, x, y, radius, B, G, R, A, randomGene):
        # Create the gene randomly if randomGene is selected or circle does not lie in the frame limits
        if randomGene or (not CheckBoundaries(x, y, radius)):
            self.coordinate = (random.randint(0, 149), random.randint(0, 222))
            self.radius = random.randint(1, 20)
            self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            self.A = random.random()

        # Create the gene with the given values
        else:
            self.coordinate = (x, y)
            self.radius = radius
            self.color = (B, G, R)
            self.A = A

        return [self.coordinate[0], self.coordinate[1], self.radius, self.color[0], self.color[1], self.color[2],
                self.A]

    # Mutate the gene
    def mutate(self, mutationMethod="Guided"):
        if mutationMethod == "Unguided":
            self.coordinate = (random.randint(0, 149), random.randint(0, 222))
            self.radius = random.randint(1, 100)
            self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            self.A = random.random()
        elif mutationMethod == "Guided":
            mutationCoordinate = (random.randint(self.coordinate[0] - 37, self.coordinate[0] + 37),
                                  random.randint(self.coordinate[1] - 64, self.coordinate[1] + 64))
            mutationRadius = random.randint(max(1, self.radius-10), self.radius+10)
            # If the circle is not in the frame range, repeat the process until it is in range
            if not CheckBoundaries(mutationCoordinate[0], mutationCoordinate[1], mutationRadius):
                return self.mutate(mutationMethod)
            self.coordinate = deepcopy(mutationCoordinate)
            self.radius = deepcopy(mutationRadius)
            self.color = (random.randint(max(0, self.color[0] - 64), min(255, self.color[0] + 64)),
                          random.randint(max(0, self.color[1] - 64), min(255, self.color[1] + 64)),
                          random.randint(max(0, self.color[2] - 64), min(255, self.color[2] + 64)))
            self.A = random.uniform(max(0, self.A - 0.25), min(1, self.A + 0.25))
        self.geneArray = [self.coordinate[0], self.coordinate[1], self.radius, self.color[0], self.color[1],
                          self.color[2], self.A]


# Define the Individual as a class
class Individual:
    def __init__(self, num_genes=50):
        self.num_genes = num_genes
        self.fitness = None
        self.chromosome = self.initializeChromosome()

    # Initialize the chromosome with the number of genes
    def initializeChromosome(self):
        self.fitness = None
        return [Gene(randomGene=True) for i in range(self.num_genes)]
        #return sorted(unsortedChromosome, key=lambda x: x.radius, reverse=True)

    # Mutate the individual
    def mutateIndividual(self, mutation_prob=0.2, mutationMethod="Guided"):
        if random.random() < mutation_prob:
            mutatedGeneIndex = random.randint(0, self.num_genes-1)
            self.chromosome[mutatedGeneIndex].mutate(mutationMethod)
            self.fitness = None
        #self.chromosome.sort(key=lambda x: x.radius, reverse=True)

    # Evaluate the individual's fitness
    def evaluateFitness(self, monaLisaImage):
        # Get the size of the Mona Lisa
        height = monaLisaImage.shape[0]  # 223
        width = monaLisaImage.shape[1]  # 150
        # Create the blank image
        img = np.full((height, width, 3), (255, 255, 255), np.uint8)
        # For each gene in the chromosome
        for gene in self.chromosome:
            # Overlay <- img
            overlay = deepcopy(img)
            # Draw the circle on overlay
            cv2.circle(overlay, gene.coordinate, gene.radius, gene.color, thickness=-1)
            # img <- overlay x alpha + img x (1 - alpha)
            img = np.add((overlay * gene.A), (img * (1 - gene.A)))
        # Calculate the fitness in a vectorized fashion to increase efficiency of the code
        self.fitness = -np.sum(np.square(np.subtract(monaLisaImage, img)))

    # Create offspring from parents
    def createOffspring(self, Parent0, Parent1, geneOwnerArray):
        for i in range(self.num_genes):
            # If gene comes from Parent0
            if geneOwnerArray[i] == 0:
                self.chromosome[i] = deepcopy(Parent0.chromosome[i])
            elif geneOwnerArray[i] == 1:
                self.chromosome[i] = deepcopy(Parent1.chromosome[i])
        #self.chromosome.sort(key=lambda x: x.radius, reverse=True)
        self.fitness = None

    def drawImage(self, monaLisaImage):
        # Get the size of the Mona Lisa
        height = monaLisaImage.shape[0]  # 223
        width = monaLisaImage.shape[1]  # 150
        # Create the blank image
        img = np.full((height, width, 3), (255, 255, 255), np.uint8)
        # For each gene in the chromosome
        for gene in self.chromosome:
            # Overlay <- img
            overlay = deepcopy(img)
            # Draw the circle on overlay
            cv2.circle(overlay, gene.coordinate, gene.radius, gene.color, thickness=-1)
            # img <- overlay x alpha + img x (1 - alpha)
            img = np.add((overlay * gene.A), (img * (1 - gene.A)))
        return img


# Define the population as a class
class Population:
    def __init__(self, num_inds=20, num_genes=50, tm_size=5, frac_elites=0.2, frac_parents=0.6, mutation_prob=0.2,
                 mutationMethod="Guided"):
        # Characteristics of a population - They do not change from generation to generation
        self.num_inds = num_inds
        self.num_genes = num_genes
        self.tm_size = tm_size
        self.frac_elites = frac_elites
        self.frac_parents = frac_parents
        self.mutation_prob = mutation_prob
        self.mutationMethod = mutationMethod

        # These values may change from generation to generation
        self.members = self.initializePopulation()
        self.elites = []
        self.tournamentWinners = []
        self.parentTournamentWinners = []
        self.advancingTournamentWinners = []
        self.children = []
        # This list holds the members that will be in the next generation
        self.nextGeneration = []

    # Initialize the population with num_inds and num_genes
    def initializePopulation(self):
        return [Individual(self.num_genes) for i in range(self.num_inds)]

    # Evaluate all members
    def evaluatePopulation(self, monaLisaImage):
        for member in self.members:
            member.evaluateFitness(monaLisaImage)

    # Find the elites
    def findElites(self):
        self.elites = sorted(self.members, key=lambda x: x.fitness, reverse=True)[0:int(self.num_inds*self.frac_elites)]

    # Tournament Selection
    def tournamentSelection(self):
        for i in range(self.num_inds - int(self.num_inds*self.frac_elites)):
            tournamentMembers = random.sample(self.members, self.tm_size)
            self.tournamentWinners.append(max(tournamentMembers, key=attrgetter("fitness")))
        # If we want to choose the parents according to their fitnesses, activate the following line
        #self.tournamentWinners.sort(key=lambda x: x.fitness, reverse=True)

    # Selection of elites and tournament winners(both parentWinners and advancingWinners)
    def selection(self):
        self.findElites()
        self.tournamentSelection()
        # Choose parentWinners and advancingWinners - Best tournament winners will be parents
        self.parentTournamentWinners = self.tournamentWinners[0:int(self.num_inds*self.frac_parents)]
        self.advancingTournamentWinners = self.tournamentWinners[int(self.num_inds*self.frac_parents):]

    # Create children with crossover
    def crossover(self):
        for i in range(0, len(self.parentTournamentWinners), 2):
            # If the number of parents are odd, the single parent advances to the next generation since no pair left
            if i == len(self.parentTournamentWinners) - 1:
                self.children.append(self.parentTournamentWinners[i])
            else:
                # Select the parents
                Parent0 = self.parentTournamentWinners[i]
                Parent1 = self.parentTournamentWinners[i+1]
                # Initialize the children individuals
                Children0 = Individual(self.num_genes)
                Children1 = Individual(self.num_genes)
                # geneOwnerArrays are binomial and each element describes which parent that gene comes from
                geneOwnerArray0 = np.random.binomial(1, 0.5, self.num_genes)
                # geneOwnerArray1 is binomial complement of geneOwnerArray0
                geneOwnerArray1 = np.ones(self.num_genes, dtype=int) - geneOwnerArray0
                # Create the children
                Children0.createOffspring(Parent0, Parent1, geneOwnerArray0)
                Children1.createOffspring(Parent0, Parent1, geneOwnerArray1)
                # Add the children to the children list
                self.children.append(Children0)
                self.children.append(Children1)

    # Next generation members are elites + children + tournament winners that didn't participate to crossover
    def nextGenerationMembers(self):
        self.nextGeneration = self.elites + self.children + self.advancingTournamentWinners

    # Mutate the population
    def mutatePopulation(self):
        for member in self.nextGeneration:
            if member not in self.elites:
                member.mutateIndividual(self.mutation_prob, self.mutationMethod)

    # Next generation becomes the current generation at the end of each epoch
    def generationUpdate(self):
        self.members = deepcopy(self.nextGeneration)
        # Reset the lists
        self.elites = []
        self.tournamentWinners = []
        self.parentTournamentWinners = []
        self.advancingTournamentWinners = []
        self.children = []
        self.nextGeneration = []

    # Find the best Individual of the population for the current generation
    def bestIndividual(self):
        bestIndividual = max(self.members, key=attrgetter("fitness"))
        return bestIndividual

    # Find the total fitness of the population for the current generation
    def totalFitness(self):
        total = 0
        for member in self.members:
            total += member.fitness
        return total

    def sortAllChromosomes(self):
        for member in self.members:
            member.chromosome.sort(key=lambda x: x.radius, reverse=True)
