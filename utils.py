import numpy as np


class HeatmapBuffer:
    
    def __init__(self, shape):
        self.shape = shape
        self.data = np.zeros(self.shape)
        self.index = 0
        self.items_added = 0
    
    def add_heatmap(self, x):
        self.items_added += 1
        self.data[self.index] = x
        self.index = (self.index + 1) % self.shape[0]
        
    def mean_heatmap(self):
        if self.items_added < self.shape[0]:
            # give an adjusted mean if insufficient data so far
            return np.mean(self.data[:self.items_added], axis=0)
        else:
            return np.mean(self.data, axis=0)
