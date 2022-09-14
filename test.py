import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def show_tensor_images(image_tensor, num_images=16, size=(3, 64, 64), nrow=3):
    '''
    Function for visualizing images: Given a tensor of images, number of images,
    size per image, and images per row, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu().clamp_(0, 1)
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow, padding=2)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    plt.show()

class ModulatedConv2d(nn.Module):
    '''
    ModulatedConv2d Class, extends/subclass of nn.Module
    Values:
      channels: the number of channels the image has, a scalar
      w_dim: the dimension of the intermediate tensor, w, a scalar 
    '''
    def __init__(self, w_dim, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.conv_weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.style_scale_transform = nn.Linear(w_dim, in_channels)
        self.eps = 1e-6
        self.padding = padding

    def forward(self, image, w):
        # There is a more efficient (vectorized) way to do this using the group parameter of F.conv2d,
        # but for simplicity and readibility you will go through one image at a time.
        images = []
        for i, w_cur in enumerate(w):
            # Calculate the style scale factor
            style_scale = self.style_scale_transform(w_cur)
            # Multiply it by the corresponding weight to get the new weights
            w_prime = self.conv_weight * style_scale[None, :, None, None]
            # Demodulate the new weights based on the above formula
            w_prime_prime = w_prime / torch.sqrt(
                (w_prime ** 2).sum([1, 2, 3])[:, None, None, None] + self.eps
            )
            images.append(F.conv2d(image[i][None], w_prime_prime, padding=self.padding))
        return torch.cat(images)
    
    def forward_efficient(self, image, w):
        # Here's the more efficient approach. It starts off mostly the same
        style_scale = self.style_scale_transform(w)
        w_prime = self.conv_weight[None] * style_scale[:, None, :, None, None]
        w_prime_prime = w_prime / torch.sqrt(
            (w_prime ** 2).sum([2, 3, 4])[:, :, None, None, None] + self.eps
        )
        # Now, the trick is that we'll make the images into one image, and 
        # all of the conv filters into one filter, and then use the "groups"
        # parameter of F.conv2d to apply them all at once
        batchsize, in_channels, height, width = image.shape
        out_channels = w_prime_prime.shape[2]
        # Create an "image" where all the channels of the images are in one sequence
        efficient_image = image.view(1, batchsize * in_channels, height, width)
        efficient_filter = w_prime_prime.view(batchsize * out_channels, in_channels, *w_prime_prime.shape[3:])
        efficient_out = F.conv2d(efficient_image, efficient_filter, padding=self.padding, groups=batchsize)
        return efficient_out.view(batchsize, out_channels, *image.shape[2:])

# For convenience, we'll define a very simple generator here:
class SimpleGenerator(nn.Module):
    '''
    SimpleGenerator Class, for path length regularization demonstration purposes
    Values:
      channels: the number of channels the image has, a scalar
      w_dim: the dimension of the intermediate tensor, w, a scalar 
    '''

    def __init__(self, w_dim, in_channels, hid_channels, out_channels, kernel_size, padding=1, init_size=64):
        super().__init__()
        self.w_dim = w_dim
        self.init_size = init_size
        self.in_channels = in_channels
        self.c1 = ModulatedConv2d(w_dim, in_channels, hid_channels, kernel_size)
        self.activation = nn.ReLU()
        self.c2 = ModulatedConv2d(w_dim, hid_channels, out_channels, kernel_size)

    def forward(self, w):
        image = torch.randn(len(w), self.in_channels, self.init_size, self.init_size).to(w.device)
        y = self.c1(image, w)
        y = self.activation(y)
        y = self.c2(y, w)
        return y

from torch.autograd import grad

def PathLengthRegulator(faked_image, latent, mean_path, w_decay):
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
        create_graph=True)[0]
    if gradient == None:
        gradient = torch.randn(faked_image.shape)
    # Calculate penalty
    penalty = (((gradient - mean_path) ** 2).sum()).sqrt()

    # new a = a * (1-w_decay) + ||Jw * y - a||(2) * w_decay
    path_length = ((gradient ** 2).sum()).sqrt()
    new_mean_path = mean_path * (1-w_decay) + w_decay * path_length
    return penalty, new_mean_path

simple_gen = SimpleGenerator(w_dim=128, in_channels=3, hid_channels=64, out_channels=3, kernel_size=3)
samples = 10
test_w = torch.randn(samples, 128).requires_grad_()
faked = simple_gen(test_w)
a = 10
penalty, new_a = PathLengthRegulator(faked, test_w, a, 0.001)

# decay = 0.001 # How quickly a should decay
# new_a = a * (1 - decay) + variation * decay
print(f"Old a: {a}; new a: {new_a.item()}")