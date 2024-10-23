import torch.cuda as cuda

def memcheck():
    return (cuda.memory_allocated(), cuda.memory_reserved())
