import numpy as np
import cv2
from classes import Population
import matplotlib.pyplot as plt


# Code piece to draw fitness plots
def drawPlots(totalFitnessList, num_generations, num_inds, num_genes, tm_size,
              frac_elites, frac_parents, mutation_prob, mutationMethod):
    generations = [i for i in range(num_generations+1)]

    plt.plot(generations, totalFitnessList)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Total Fitness vs All Generations")
    plt.savefig("Results/Plots/first_"+str(num_inds)+"_"+str(num_genes)+"_"+str(tm_size)+"_"+str(frac_elites)+
                "_"+str(frac_parents)+"_"+str(mutation_prob)+"_"+mutationMethod+".png")
    plt.close()

    plt.plot(generations[1000:], totalFitnessList[1000:])
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Total Fitness vs Generations 1000-10000")
    plt.savefig("Results/Plots/second_"+str(num_inds)+"_"+str(num_genes)+"_"+str(tm_size)+"_"+str(frac_elites)+
                "_"+str(frac_parents)+"_"+str(mutation_prob)+"_"+mutationMethod+".png")
    plt.close()


# Hyperparameters
num_generations = 10000
num_inds = 20
num_genes = 50
tm_size = 5
frac_elites = 0.2
frac_parents = 0.6
mutation_prob = 0.2
mutationMethod = "Guided"

# Read the Mona Lisa image
monaLisaImage = cv2.imread("mona_lisa.jpg")
generationNumber = 0
totalFitnessList = []

# EVOLUTIONARY ALGORITHM
# Initialize population with num_inds individuals each having num_genes genes
pop = Population(num_inds=num_inds, num_genes=num_genes, tm_size=tm_size,
                 frac_elites=frac_elites, frac_parents=frac_parents, mutation_prob=mutation_prob,
                 mutationMethod="Guided")
# While not all generations (num_generations) are computed:
while generationNumber <= num_generations:
    # Sort all chromosomes before evaluating
    pop.sortAllChromosomes()
    # Evaluate all the individuals
    pop.evaluatePopulation(monaLisaImage=monaLisaImage)
    # Find best individual and save total fitness
    bestInd = pop.bestIndividual()
    totalFit = pop.totalFitness()
    totalFitnessList.append(totalFit)
    print(bestInd.fitness, generationNumber)

    # Save the created image
    if generationNumber % 1000 == 0:
        img = bestInd.drawImage(monaLisaImage)
        cv2.imwrite("Results/Mona/" +str(generationNumber)+"_"+str(num_inds)+"_"+str(num_genes)+"_"+str(tm_size)+"_"+
                    str(frac_elites)+"_"+str(frac_parents)+"_"+str(mutation_prob)+"_"+mutationMethod+".jpg", img)

    # Select individuals
    pop.selection()
    # Do crossover on some individuals
    pop.crossover()
    # Create next generation list
    pop.nextGenerationMembers()
    # Mutate some individuals
    pop.mutatePopulation()
    # Update the population
    pop.generationUpdate()
    # Increase the generation number
    generationNumber += 1

drawPlots(totalFitnessList, num_generations, num_inds, num_genes, tm_size, frac_elites,
          frac_parents, mutation_prob, mutationMethod)
