import torch

GAIN = 1
LRM = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 2e-2

Z_DIM = 256
W_DIM = 128
BATCH_SIZE = 64
