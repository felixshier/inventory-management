# imports
import numpy as np
import scipy.stats as ss

# functions

# quantization function

def quantize(val, binSize):
    newVal = binSize * np.round(val/binSize)
    return int(newVal)

def holdingCost(stock, orders):

    # fixed holding cost
    h = 5

    # total holding cost
    holdingCost = h * (stock + orders)

    return holdingCost

def orderCost(orders):
    
    # single order cost
    c = 30

    # order cost
    orderCost = c * orders

    return orderCost

def revenue(stock, orders, demand):

    # price
    p = 100

    # revenue
    revenue = p * np.minimum(stock + orders, demand)

    return revenue

def costFunction(stock, orders, demand):

    costFunction = holdingCost(stock, orders) + orderCost(orders) - revenue(stock, orders, demand)
    
    return costFunction

def createNormalDemand(myclip_a, myclip_b, my_mean, my_std):
    # define distribution
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    demand = ss.truncnorm(a, b, loc = my_mean, scale = my_std)
    x_range = np.linspace(0,capacity,capacity)
    pmf = ss.truncnorm.pdf(x_range, a, b, loc = my_mean, scale = my_std)
    pmf = [prob/sum(pmf) for prob in pmf]
    values = (x_range, pmf)
    demand = ss.rv_discrete(values = values)

    return demand

