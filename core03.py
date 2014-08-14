### ludobots core03
### Riley Pinkerton 8/14/2014

import random
import math

import numpy as np
import matplotlib.pyplot as pp

from core01 import matrix_create, matrix_randomize
from core02 import update

NEURON_GOAL = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]

def dist(v1, v2):
  """Returns the distance between v1 and v2 in n-space"""
  if len(v1) != len(v2):
    print("Dimension mismatch!")
    print v1
    print v2
    exit(1)
  d_squared = 0.0
  for i in range(len(v1)):
    d_squared += (v2[i] - v1[i]) ** 2
  return math.sqrt(d_squared)

def nn_fitness(synapses, graph=False):
  """Returns the fitness of a set of neural network synapses. Shows a matrix
     graph if graph is true. This fitness tries to reach a checkerboard pattern
  """
  # Determine fitness by running the synapse values on a neural network
  neuron_values = matrix_create(10, 10)
  for i in range(10):
    neuron_values[0, i] = 0.5
  for i in range(9):
    update(neuron_values, synapses, i)
  dist_to_goal = dist(NEURON_GOAL, neuron_values[9, :].A.reshape(-1).tolist())
  # Since neuron values are in [0, 1], max distance is sqrt(10)
  norm_dist = dist_to_goal / math.sqrt(10)
  if graph:
    pp.imshow(neuron_values, cmap=pp.cm.gray, aspect='auto',
              interpolation='nearest')
    pp.show()
  return 1.0 - norm_dist

def nn_fitness2(synapses, graph=False):
  """Returns the fitness of a set of neural network synapses. Shows a matrix
     graph if graph is true
  """
  # Determine fitness by running the synapse values on a neural network
  neuron_values = matrix_create(10, 10)
  for i in range(10):
    neuron_values[0, i] = 0.5
  for i in range(9):
    update(neuron_values, synapses, i)
  diff = 0.0
  for i in range(0,9):
    for j in range(0,9):
       diff += abs(neuron_values[i, j] - neuron_values[i, j + 1])
       diff += abs(neuron_values[i + 1, j] - neuron_values[i, j])
  if graph:
    pp.imshow(neuron_values, cmap=pp.cm.gray, aspect='auto',
              interpolation='nearest')
    pp.show()
  return diff / 162.0

def nn_perturb(synapses, probability):
    """Returns a perturbed copy of synapses, depending on probability"""
    new_mat = synapses.copy()
    x, y = new_mat.shape
    for i in range(x):
        for j in range(y):
            if probability > random.random():
                new_mat[i, j] = random.random() * 2 - 1
    return new_mat

def plot_nn(fitness):
    """Plots a graph of fitness for 1000 generations, and gives a matrix display
       visualizing the generations, using a given fitness function
    """
    parent = matrix_create(10, 10)
    matrix_randomize(parent)
    # Map random values into [-1, 1]
    for i in range(10):
      for j in range(10):
        parent[i, j] *= 2
        parent[i, j] -= 1
    parent_fitness = fitness(parent, graph=True)
    fitness_array = []

    for gen in range(0, 1000):
        fitness_array.append(parent_fitness)
        child = nn_perturb(parent, 0.05)
        child_fitness = fitness(child)

        if child_fitness > parent_fitness:
            parent = child
            parent_fitness = child_fitness
    final_fitness = fitness(parent, graph=True)
    pp.plot(fitness_array)

if __name__ == "__main__":
  plot_nn(nn_fitness)
  pp.show()

  plot_nn(nn_fitness2)
  pp.show()
