import copy
import numpy as np
from typing import Callable

class Px:
    def __init__(self, dx: int, dy: int):
        self.dist = np.array([dx, dy])

    def dist_sq(self):
        return np.linalg.norm(self.dist)

    def __le__(self, other):
        assert(isinstance(other, Px))
        return self.dist_sq() <= other.dist_sq()

    def __lt__(self, other):
        assert (isinstance(other, Px))
        return self.dist_sq() < other.dist_sq()

    def __sub__(self, other):
        assert (isinstance(other, Px))
        return self.dist - other.dist


class Grid:
    def __init__(self, w: int = 100, h: int = 100):
        self.width = w
        self.height = h
        self.grids = [np.array([[Px(0,0)]*w]*h)]*2
        self.sdt = None

    def add_object(self, constraint: Callable[[int, int], bool]):
        for x in range(self.width):
            for y in range(self.height):
                if constraint(x, y):
                    self.grids[0][x, y] = Px(0,0)
                    self.grids[1][x, y] = Px(np.inf, np.inf)
                else:
                    self.grids[0][x, y] = Px(np.inf, np.inf)
                    self.grids[1][x, y] = Px(0,0)

    def _compare(self, grid, x: int, y: int, offset_x: int, offset_y: int):
        neighbor =  copy.deepcopy(self.grids[grid][x+offset_x,y+offset_y])
        neighbor.dist += np.array([offset_x,offset_y])

        if neighbor < self.grids[grid][x, y]:
            self.grids[grid][x,y] = neighbor

    def generateSDT(self):
        y_range = list(range(1, self.height-1))
        x_range = list(range(1, self.width-1))
        for i in range(2):
            for y in y_range:
                print("y = {}".format(y))
                for x in x_range:
                    self._compare(i, x, y, -1,  0)
                    self._compare(i, x, y,  0, -1)
                    self._compare(i, x, y, -1, -1)
                    self._compare(i, x, y,  1, -1)
                x_range.reverse()
                for x in x_range:
                    self._compare(i, x, y,  1,  0)

            y_range.reverse()
            x_range.reverse()
            for y in y_range:
                for x in x_range:
                    self._compare(i, x, y,  1,  0)
                    self._compare(i, x, y,  0,  1)
                    self._compare(i, x, y, -1,  1)
                    self._compare(i, x, y,  1,  1)
                x_range.reverse()                    
                for x in x_range:
                    self._compare(i, x, y,  -1,  0)
            y_range.reverse()
            x_range.reverse()

        self.sdt = self.grids[0] - self.grids[1]

if __name__ == "__main__":
    grid = Grid()
    grid.add_object(lambda x, y: 30<x<50 and 30<y<50)
    grid.generateSDT()
    print("sdt done")
    print(grid.sdt)