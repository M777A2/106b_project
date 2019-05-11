from DistanceField import Grid
from scipy.optimize import minimize
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np
from itertools import permutations

grid = Grid(70, 70)
grid.add_object(lambda x, y: 30 < x < 50 and 30 < y < 40)
grid.generateSDT()
print(grid)
field = np.array(grid.get())

initial = (40, 20)
goal = (36, 50)

# path = [[40, 20, 0, 0]] * 100
path = [initial, goal]
waypoints = sig.resample(path, 10)

x0 = [initial, goal]


def cost(x):
    return sum(map(lambda x, y: -1 * field[x, y], x)) + \
           sum([np.linalg.norm(x[i + 1]) - np.linalg.norm(x[i]) for i in range(len(x) - 1)])


def grad(x):
    x.reshape(2, len(x)//2)
    der = np.zeros_like(x)
    around = list(set(permutations([-1, -1, 0, 1, 1], 2)))
    for i, dx in enumerate(x):
        der[i] = max(around, key=lambda j: field[j[0]][j[1]])
        der[i] = np.array(der[i]) * 2 + dx
    return der


res = minimize(cost, x0, method="BFGS", jac=grad, options={'disp': True})

print(res.x)