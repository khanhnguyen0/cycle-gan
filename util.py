import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def weights_init(m :nn.Module):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def show_tensor_images(image_tensor, writer, num_images=25, size=(1, 28, 28), step=0, tag=''):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    # plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    # plt.show()
    writer.add_image(tag, image_grid, step)