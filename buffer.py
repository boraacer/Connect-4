import random

# class to store the memory of the neural network for training
class ReplayBuffer(object):
    def __init__(self):
        # memory is an array of transition vectors
        self.mem = []
    
    # method to add a transition vector to the memory
    def add(self, transition_vector):
        self.mem.append(transition_vector)
    
    # sample a number of transition vectors from the memory
    def sample(self, batch_size):
        return random.sample(self.mem, batch_size)
    
    # method to get the size of the memory
    def __len__(self):
        return len(self.mem)
