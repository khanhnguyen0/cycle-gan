import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from skimage import color
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from settings import *
from loss import get_disc_loss, get_gen_loss
from util import show_tensor_images

from data.dataset import ImageDataset
from networks.generator import Generator
from networks.discriminator import Discriminator




def train(save_model=True):
    adv_criterion = nn.MSELoss() 
    recon_criterion = nn.L1Loss() 
    transform = transforms.Compose([
    transforms.Resize(LOAD_SHAPE),
    transforms.RandomCrop(TARGET_SHAPE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])

    dataset = ImageDataset("horse2zebra", transform=transform)

    gen_AB = Generator(DIM_A, DIM_B).to(DEVICE)
    gen_BA = Generator(DIM_B, DIM_A).to(DEVICE)
    gen_opt = torch.optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=LR, betas=(0.5, 0.999))
    disc_A = Discriminator(DIM_A).to(DEVICE)
    disc_A_opt = torch.optim.Adam(disc_A.parameters(), lr=LR, betas=(0.5, 0.999))
    disc_B = Discriminator(DIM_B).to(DEVICE)
    disc_B_opt = torch.optim.Adam(disc_B.parameters(), lr=LR, betas=(0.5, 0.999))

    plt.rcParams["figure.figsize"] = (10, 10)
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    cur_step = 0

    writer = SummaryWriter()


    for epoch in range(N_EPOCHS):
        # Dataloader returns the batches
        # for image, _ in tqdm(dataloader):
        for real_A, real_B in tqdm(dataloader):
            # image_width = image.shape[3]
            real_A = nn.functional.interpolate(real_A, size=TARGET_SHAPE)
            real_B = nn.functional.interpolate(real_B, size=TARGET_SHAPE)
            cur_batch_size = len(real_A)
            real_A = real_A.to(DEVICE)
            real_B = real_B.to(DEVICE)

            ### Update discriminator A ###
            disc_A_opt.zero_grad() # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake_A = gen_BA(real_B)
            disc_A_loss = get_disc_loss(real_A, fake_A, disc_A, adv_criterion)
            disc_A_loss.backward(retain_graph=True) # Update gradients
            disc_A_opt.step() # Update optimizer

            ### Update discriminator B ###
            disc_B_opt.zero_grad() # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake_B = gen_AB(real_A)
            disc_B_loss = get_disc_loss(real_B, fake_B, disc_B, adv_criterion)
            disc_B_loss.backward(retain_graph=True) # Update gradients
            disc_B_opt.step() # Update optimizer

            ### Update generator ###
            gen_opt.zero_grad()
            gen_loss, fake_A, fake_B = get_gen_loss(
                real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, recon_criterion, recon_criterion
            )
            gen_loss.backward() # Update gradients
            gen_opt.step() # Update optimizer

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_A_loss.item() / DISPLAY_STEP
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / DISPLAY_STEP

            # Add losses to tensorboard
            writer.add_scalar('gen_loss', gen_loss.item(), cur_step)
            writer.add_scalar('disc_A_loss', disc_A_loss.item(), cur_step)
            writer.add_scalar('disc_A_loss', disc_B_loss.item(), cur_step)



            ### Visualization code ###
            if cur_step % DISPLAY_STEP == 0:
                print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                show_tensor_images(torch.cat([real_A, real_B]), writer=writer, size=(DIM_A, TARGET_SHAPE, TARGET_SHAPE), step=cur_step, tag='AtoB')
                show_tensor_images(torch.cat([fake_B, fake_A]), writer=writer, size=(DIM_B, TARGET_SHAPE, TARGET_SHAPE), step=cur_step, tag='BtoA')
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                # You can change save_model to True if you'd like to save the model
                if save_model:
                    torch.save({
                        'gen_AB': gen_AB.state_dict(),
                        'gen_BA': gen_BA.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc_A': disc_A.state_dict(),
                        'disc_A_opt': disc_A_opt.state_dict(),
                        'disc_B': disc_B.state_dict(),
                        'disc_B_opt': disc_B_opt.state_dict()
                    }, f"cycleGAN_{cur_step}.pth")
            cur_step += 1
train()