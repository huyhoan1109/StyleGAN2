import math
from torch.autograd import grad, Variable
import torch

from config import WEIGHT_DECAY

def PLRegulator(
    # Path length Regulator
    faked_image:torch.Tensor, 
    latent:torch.Tensor, 
    mean_path:float, 
    w_decay:float):
    # faked_images => Generated noise y
    y = torch.randn(faked_image.shape) /  math.sqrt(
        faked_image.shape[2] * faked_image.shape[3]
    )
    # output = g(w) * y
    output = (faked_image * y).sum()
    # gradient = Jw * y 
    gradient = grad(
        outputs=output, 
        inputs=latent, 
        allow_unused=True, 
        create_graph=True)[0]
    if gradient == None:
        raise TypeError("Can't calculate gradient!")
    # Calculate penalty
    penalty = (((gradient - mean_path) ** 2).sum()).sqrt()
    # new a = a * (1-w_decay) + ||Jw * y - a||(2) * w_decay
    path_length = ((gradient ** 2).sum()).sqrt()
    new_mean_path = mean_path * (1-w_decay) + w_decay * path_length
    return penalty, new_mean_path

faked = torch.rand(5, 3, 64, 64)
latent = torch.rand(5, 128)
mean_path = 10
penalty, new_mean_path = PLRegulator(faked, latent, mean_path, WEIGHT_DECAY)
print(penalty, new_mean_path)