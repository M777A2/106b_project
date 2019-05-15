import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import norm
from typing import Callable, Iterable


class Grid:
    def __init__(self, w: int = 100, h: int = 100):
        self.width = w
        self.height = h
        self.MAX = w**2 + h**2
        self.grids = [np.array([[np.zeros(2)]*w]*h)]*2
        self.grids = [np.array([[np.zeros(2)]*w]*h), np.array([[np.ones(2)*10]*w]*h)]
        self.sdt = None

    def _show_grid(self, show) -> str:
        if show is None:
            return ""
        if isinstance(show[0, 0], Iterable):
            foo = lambda x, y: " "*(3-len(str(int(norm(show[x, y]))))) + str(int(norm(show[x, y])))
        else:
            foo = lambda x, y: " "*(3-len(str(int(show[x, y])))) + str(int(show[x, y]))
        rep = ""
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                line += foo(x, y)
            rep += line + "\n"
        return rep

    def __repr__(self):
        return self._show_grid(self.sdt)

    def __str__(self):
        return self.__repr__()

    def add_object(self, constraint: Callable[[int, int], bool]):
        for x in range(self.width):
            for y in range(self.height):
                if constraint(x, y):
                    self.grids[0][x, y] = np.array([0, 0])
                    self.grids[1][x, y] = np.array([self.MAX, self.MAX])
                else:
                    self.grids[0][x, y] = np.array([self.MAX, self.MAX])
                    self.grids[1][x, y] = np.array([0, 0])

        # for grid in range(2):
        #     print("grid: {}".format(grid))
        #     print(self._show_grid(self.grids[grid]))

    def _compare(self, grid, x: int, y: int, offset_x: int, offset_y: int):
        if 0 <= (x+offset_x) < self.width and 0 <= (y+offset_y) < self.height:
            neighbor = copy.deepcopy(self.grids[grid][x+offset_x, y+offset_y])
            neighbor += np.array([offset_x, offset_y])
        else:
            neighbor = np.array([self.MAX, self.MAX])

        if sum(neighbor**2) < sum(self.grids[grid][x, y]**2):
            # print("changing value [{}, {}]".format(x, y))
            # print("grid: {}".format(grid))
            # print(self._show_grid(self.grids[grid]))
            self.grids[grid][x, y] = neighbor

    def generateSDT(self):
        y_range = list(range(self.height))
        y_rev = list(y_range)
        y_rev.reverse()
        x_range = list(range(self.width))
        x_rev = list(x_range)
        x_rev.reverse()

        for i in range(2):  # compute for each grid separately
            for y in y_range:
                for x in x_range:
                    self._compare(i, x, y, -1,  0)
                    self._compare(i, x, y,  0, -1)
                    self._compare(i, x, y, -1, -1)
                    self._compare(i, x, y,  1, -1)
                for x in x_rev:
                    self._compare(i, x, y,  1,  0)

            for y in y_rev:
                for x in x_rev:
                    self._compare(i, x, y,  1,  0)
                    self._compare(i, x, y,  0,  1)
                    self._compare(i, x, y, -1,  1)
                    self._compare(i, x, y,  1,  1)
                for x in x_range:
                    self._compare(i, x, y, -1,  0)

        self.sdt = np.zeros([self.width, self.height])
        for y in range(self.height):
            for x in range(self.width):
                self.sdt[x, y] = norm(self.grids[0][x, y]) - norm(self.grids[1][x, y])


    def get(self) -> np.array:
        return self.sdt


if __name__ == "__main__":
    size = 80
    grid = Grid(size, size)
    grid.add_object(lambda x, y: 30 < x < 60 and 30 < y < 60)
    grid.generateSDT()
    print("sdt done")
    print(grid)

    # Import Dataset
    df = grid.get()

    # Plot
    plt.figure(figsize=(size, size), dpi=9)
    sns.heatmap(df, xticklabels=range(size), yticklabels=range(size), cmap='RdYlGn', center=0,
                annot=False)

    # Decorations
    plt.title('Signed Distance Field', fontsize=120)
    plt.xticks(fontsize=62)
    plt.yticks(fontsize=62)
    plt.show()
