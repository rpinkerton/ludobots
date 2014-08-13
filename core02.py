### ludobots core02
### Riley Pinkerton 8/12/2014

import random
import math

import numpy as np
import matplotlib.pyplot as pp

from core01 import *

def get_neuron_positions(num):
    """Creates an array of neuron positions for num neurons"""
    positions = matrix_create(2, num)
    angle = 0.0
    delta = 2.0 * math.pi / float(num)
    for i in range(num):
        positions[0, i] = math.sin(angle)
        positions[1, i] = math.cos(angle)
        angle += delta
    return positions

def plot_all_lines(positions):
    """Plots a complete graph over the positions"""
    for i in range(10):
        for j in [x for x in range(10) if x != i]:
            pp.plot([neuron_positions[0, i], neuron_positions[0, j]],
                    [neuron_positions[1, i], neuron_positions[1, j]],
                    color=[0,0,0])
    # Still need to plot circles again...
    for i in range(10):
        pp.plot(neuron_positions[0, i], neuron_positions[1, i],
                'ko', markerfacecolor=[1,1,1], markersize=18)
    pp.show()

def plot_colored_lines(positions, synapses):
    """Plots a complete graph over the positions with colors for synapse
       signs
    """
    for i in range(10):
        for j in [x for x in range(10) if x != i]:
            if synapses[i, j] < 0:
                line_color = [0.8, 0.8, 0.8]
            else:
                line_color = [0, 0, 0]
            pp.plot([neuron_positions[0, i], neuron_positions[0, j]],
                    [neuron_positions[1, i], neuron_positions[1, j]],
                    color=line_color)
    # Still need to plot circles again...
    for i in range(10):
        pp.plot(neuron_positions[0, i], neuron_positions[1, i],
                'ko', markerfacecolor=[1,1,1], markersize=18)
    pp.show()

def plot_weighted_lines(positions, synapses):
    """Plots a complete graph over the positions with line thickness for synapse
       values
    """
    for i in range(10):
        for j in [x for x in range(10) if x != i]:
            if synapses[i, j] < 0:
                line_color = [0.8, 0.8, 0.8]
            else:
                line_color = [0, 0, 0]
            line_weight = int(10 * abs(synapses[i, j])) + 1
            pp.plot([neuron_positions[0, i], neuron_positions[0, j]],
                    [neuron_positions[1, i], neuron_positions[1, j]],
                    color=line_color, linewidth=line_weight)
    # Still need to plot circles again...
    for i in range(10):
        pp.plot(neuron_positions[0, i], neuron_positions[1, i],
                'ko', markerfacecolor=[1,1,1], markersize=18)
    pp.show()

def update(values, synapses, i):
    """Updates the neuron_values for the ith time"""
    for j in range(10):
        new_val = 0
        for k in range(10):
            new_val += synapses[j, k] * values[i, k]
        if new_val < 0:
            new_val = 0
        elif new_val > 1:
            new_val = 1
        values[i + 1, j] = new_val

if __name__ == "__main__":
    neuron_values = matrix_create(50, 10)
    for i in range(10):
        neuron_values[0, i] = random.random()
    neuron_positions = get_neuron_positions(10)
    for i in range(10):
        pp.plot(neuron_positions[0, i], neuron_positions[1, i],
                'ko', markerfacecolor=[1,1,1], markersize=18)
    pp.show()
    
    plot_all_lines(neuron_positions)
    
    synapses = matrix_create(10, 10)
    for i in range(10):
        for j in range(10):
            synapses[i, j] = random.random() * 2.0 - 1.0
    plot_colored_lines(neuron_positions, synapses)

    plot_weighted_lines(neuron_positions, synapses)

    for i in range(49):
        update(neuron_values, synapses, i)
    pp.imshow(neuron_values, cmap=pp.cm.gray, aspect='auto', interpolation='nearest')
    pp.xlabel('Neuron')
    pp.ylabel('Time Step')
    pp.show()
    
