from DistanceField import Grid
from scipy.optimize import minimize
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np
from itertools import permutations


class Optimize:
    def __init__(self, x, y, comp):
        self.grid = Grid(x, y)
        self.grid.add_object(comp)
        self.grid.generateSDT()
        self.field = np.array(self.grid.get())
        self.x0 = None

    def cost(self, u):
        try:
            return sum([np.linalg.norm(u[i + 1]) - np.linalg.norm(u[i]) for i in range(len(u) - 1)]) +\
               sum([min(self.field[int(round(u[i])), int(round(u[i+1]))], 0.1) for i in range(0, len(u), 2)])
        except IndexError:
            return 1000

    def grad(self, x):
        der = np.zeros_like(x)
        around = list(set(permutations([-1, -1, 0, 1, 1], 2)))

        for i in range(0, len(x), 2):
            def sign(j):
                x_off = round(x[i])+j[0]
                y_off = round(x[i+1])+j[1]
                return self.field[int(x_off), int(y_off)]
            der[i:i+2] = max(around, key=sign)
            der[i:i+2] = [der[i] * 2 + x[i]] + [der[i+1] * 2 + x[i+1]]
        return der

    def set_init(self, x0, n):
        if n < 2:
            self.x0 = x0[-1]
        else:
            self.x0 = sig.resample(x0, n)

        for i in range(len(x0)):
            x0[i] *= 10
            x0[i] += np.array([25, 10])

    def get_res(self):
        res = minimize(self.cost, self.x0, method="Nelder-Mead", options={'fatol': 0.1, 'disp': True})
        for i in range(len(res.x)):
            res.x[i] -= 25 if i % 2 else 10
            res.x *= 0.1
        return res

if __name__ == "__main__":
    opt = Optimize(100, 100, lambda x, y: 30 < x < 50 and 30 < y < 40)


    initial = (40, 20)
    goal = (36, 50)

    # path = [[40, 20, 0, 0]] * 100
    path = [initial, goal]
    waypoints = sig.resample(path, 10)
