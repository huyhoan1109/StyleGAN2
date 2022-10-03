import torch

GAIN = 1
LRM = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 2e-2

Z_DIM = 256
W_DIM = 128
BATCH_SIZE = 64
CHANNEL_MULTI = 2

CHANNELS = {
    4: 512,
    8: 512, 
    16: 512,
    32: 512,
    64: 256 * CHANNEL_MULTI,
    128: 128 * CHANNEL_MULTI,
    256: 64 * CHANNEL_MULTI,
    512: 32 * CHANNEL_MULTI,
    1024: 16 * CHANNEL_MULTI 
}
