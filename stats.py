from math import sqrt

class RunningStats(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.mean = None
        self.S = 0
        self.k = 0

    def push(self, x):
        self.k += 1
        if self.mean is None:
            self.mean = x
        else:
            delta = x - self.mean
            self.mean = self.mean + delta / self.k
            self.S = self.S + delta * (x - self.mean)

    @property
    def variance(self):
        if self.k > 1:
            return self.S / (self.k - 1)
        else:
            return 0

    @property
    def standard_deviation(self):
        return sqrt(self.variance)