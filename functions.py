# imports
import numpy as np
import scipy.stats as ss

# functions

# quantization function

def quantize(val, binSize):
    newVal = binSize * np.round(val/binSize)
    return newVal

def holdingCost(stock, orders, demand):

    # holding cost
    h = 1

    # shortage cost
    s = 1
    
    # holding/shortage cost
    holdingCost = s * np.maximum(0, -(stock + orders - demand)) + h * np.maximum(0, stock + orders - demand)

    return holdingCost

def orderCost(orders):
    
    # single order cost
    c = 10

    # order cost
    orderCost = c * orders

    return orderCost

def revenue(stock, orders, demand):

    # price
    p = 15

    # revenue
    revenue = p * np.minimum(stock + orders, demand)

    return revenue

def costFunction(stock, orders, demand):

    costFunction = holdingCost(stock, orders, demand) + orderCost(orders) - revenue(stock, orders, demand)
    
    return costFunction

def DP_costFunction(stock, iteration = 0):

    # depth
    N = 1

    # inventory capacity
    capacity = 200

    # define distribution
    myclip_a = 0
    myclip_b = capacity
    my_mean = capacity / 2
    my_std = np.round(capacity / 3)

    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

    demand = ss.truncnorm(a, b, loc = my_mean, scale = my_std)
    x_range = np.linspace(0,capacity,capacity)
    pmf = ss.truncnorm.pdf(x_range, a, b, loc = my_mean, scale = my_std)
    pmf = [prob/sum(pmf) for prob in pmf]
    values = (x_range, pmf)
    demand = ss.rv_discrete(values = values)

    possibleOrders = [i for i in range(capacity+1)]

    OptimalCosts = [0]

    for orders in possibleOrders:
        
        func1 = lambda x: costFunction(stock, orders, x)
        func2 = lambda x: DP_costFunction(stock + orders - x, iteration = iteration + 1)[0] if iteration < N else 0
        expectedCost = demand.expect(func1) + demand.expect(func2)
        expectedCosts.append(expectedCost)

    cost = np.min(expectedCosts)
    optimalOrders = np.argmin(expectedCosts)

    return cost, optimalOrders