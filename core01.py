### ludobots core01
### Riley Pinkerton 8/11/2014

import random

import numpy as np
import matplotlib.pyplot as pp

def matrix_create(x, y):
    """Makes an x by y matrix of all 0's"""
    list = []
    matrix = []
    for i in range(y):
        list.append(0.0)
    for i in range(x):
        matrix.append(list)
    return np.matrix(matrix)

def matrix_randomize(matrix):
    """Randomizes the elements of a matrix"""
    x, y = matrix.shape
    for i in range(x):
        for j in range(y):
            matrix.A[i][j] = random.random()

def fitness(matrix):
    """Returns the sum of all the values in matrix"""
    x, y = matrix.shape
    sum = 0
    for i in range(x):
        for j in range(y):
            sum += matrix.A[i][j]
    return float(sum) / float(x * y)

def matrix_perturb(matrix, probability):
    """Returns a slightly perturbed copy of matrix, depending on probability"""
    new_mat = matrix.copy()
    x, y = new_mat.shape
    for i in range(x):
        for j in range(y):
            if probability > random.random():
                new_mat.A[i][j] = random.random()
    return new_mat

def plot_vector_as_line(genes):
    """Plots a graph of fitness for 5000 generations, and gives a matrix display
       visualizing the generations
    """
    parent = matrix_create(1, 50)
    matrix_randomize(parent)
    parent_fitness = fitness(parent)
    fitness_array = []

    for gen in range(0, 5000):
        fitness_array.append(parent_fitness)
        for i in range(50):
            genes.A[i][gen] = parent.A[0][i]
        child = matrix_perturb(parent, 0.05)
        child_fitness = fitness(child)

        if child_fitness > parent_fitness:
            parent = child
            parent_fitness = child_fitness
    pp.plot(fitness_array)

if __name__ == "__main__":
    genes = matrix_create(50, 5000)
    plot_vector_as_line(genes)
    pp.xlabel('Generation')
    pp.ylabel('Fitness')
    pp.show()
    for i in range(5):
        plot_vector_as_line(genes)
    pp.xlabel('Generation')
    pp.ylabel('Fitness')
    pp.show()
    pp.imshow(genes, cmap=pp.cm.gray, aspect='auto', interpolation='nearest')
    pp.xlabel('Generation')
    pp.ylabel('Gene')
    pp.show()
