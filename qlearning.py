import numpy as np
import random
from functions import *

def generateQtable(capacity, demand, alpha, gamma, epsilon, iterations, bin_size):

    # initialize spaces
    state_space = [int(i * bin_size) for i in range(int(np.ceil(capacity/bin_size)) + 1)]
    action_space = [i for i in range(capacity + 1)]
    noise_space = demand

    # initialize q table
    q_table = np.zeros([len(state_space), len(action_space)])

    # initialize state
    state = random.choice(action_space)

    # perform iterations
    for i in range(iterations):

        # choose action

        if random.uniform(0,1) < epsilon:
            action = random.choice(action_space)
        else:
            action = np.argmin(q_table[state])

        # get stochastic noise
        noise = random.choice(noise_space)

        # calculate cost
        cost = costFunction(state, action, noise)

        # get next state
        if state + action - noise < 0:
            next_state = 0
        elif state + action - noise > capacity:
            next_state = capacity
        else:
            next_state = state + action - noise
        
        next_state = quantize(next_state, bin_size)

        # estimate optimal future value
        next_min = np.min(q_table[state_space.index(next_state)])

        # get old value
        old_value = q_table[state_space.index(state), action]

        # calculate new value
        new_value = ((1 - alpha) * old_value) + alpha * (cost + (gamma * next_min))

        # save new value
        q_table[state_space.index(state), action] = new_value

        # go to next state
        state = next_state

    return q_table

def generateQpolicies(q_table):

    # turn to np array
    q_table = np.array(q_table)

    # get policies
    q_policies = q_table.argmin(axis = 1)

    return q_policies