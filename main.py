from functions import *
from qlearning import *

capacity = 200
alpha = 0.1
gamma = 0.5
epsilon = 1
iterations = 1000000
mu, sigma = 100, 30
demand = np.round(np.random.normal(mu, sigma, 1000)).astype(np.int32)
bin_size = 25

q_table = generateQtable(capacity, demand, alpha, gamma, epsilon, iterations, bin_size)

q_policies = generateQpolicies(q_table)

print(q_table)

print(q_policies)