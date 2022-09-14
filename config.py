import torch

GAIN = 1
LRM = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'