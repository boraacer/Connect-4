import random

class ReplayBuffer(object):
    def __init__(self):
        self.mem = []
    
    def add(self, transition_vector):
        self.mem.append(transition_vector)
    
    def sample(self, batch_size):
        return random.sample(self.mem, batch_size)
    
    def __len__(self):
        return len(self.mem)
