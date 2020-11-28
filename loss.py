import torch
import torch.nn as nn
from typing import Tuple

def get_disc_loss(real_X: torch.Tensor, fake_X: torch.Tensor, disc_X: torch.Tensor, adv_criterion: torch.Tensor) -> torch.Tensor:
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        real_X: the real images from pile X
        fake_X: the generated images of class X
        disc_X: the discriminator for class X; takes images and returns real/fake class X
            prediction matrices
        adv_criterion: the adversarial loss function; takes the discriminator 
            predictions and the target labels and returns a adversarial 
            loss (which you aim to minimize)
    '''
    real_pred = disc_X(real_X)
    real_loss = adv_criterion(real_pred, torch.ones_like(real_pred))
    fake_pred = disc_X(fake_X.detach())
    fake_loss = adv_criterion(fake_pred, torch.zeros_like(fake_pred))
    disc_loss = (real_loss + fake_loss)/2
    return disc_loss


def get_gen_adversarial_loss(real_X: torch.Tensor, disc_Y: torch.Tensor, gen_XY: torch.Tensor, adv_criterion: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Return the adversarial loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        disc_Y: the discriminator for class Y; takes images and returns real/fake class Y
            prediction matrices
        gen_XY: the generator for class X to Y; takes images and returns the images 
            transformed to class Y
        adv_criterion: the adversarial loss function; takes the discriminator 
                  predictions and the target labels and returns a adversarial 
                  loss (which you aim to minimize)
    '''
    fake_Y = gen_XY(real_X)
    fake_Y_pred = disc_Y(fake_Y.detach())
    adversarial_loss = adv_criterion(fake_Y_pred, torch.ones_like(fake_Y_pred))
    return adversarial_loss, fake_Y


def get_identity_loss(real_X: torch.Tensor, gen_YX: nn.Module, identity_criterion: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Return the identity loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        gen_YX: the generator for class Y to X; takes images and returns the images 
            transformed to class X
        identity_criterion: the identity loss function; takes the real images from X and
                        those images put through a Y->X generator and returns the identity 
                        loss (which you aim to minimize)
    '''
    #### START CODE HERE ####
    identity_X = gen_YX(real_X)
    identity_loss = identity_criterion(real_X, identity_X)
    #### END CODE HERE ####
    return identity_loss, identity_X


def get_cycle_consistency_loss(real_X: torch.Tensor, fake_Y: torch.Tensor, gen_YX: nn.Module, cycle_criterion: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Return the cycle consistency loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        fake_Y: the generated images of class Y
        gen_YX: the generator for class Y to X; takes images and returns the images 
            transformed to class X
        cycle_criterion: the cycle consistency loss function; takes the real images from X and
                        those images put through a X->Y generator and then Y->X generator
                        and returns the cycle consistency loss (which you aim to minimize)
    '''
    #### START CODE HERE ####
    cycle_X = gen_YX(fake_Y.detach())
    cycle_loss = cycle_criterion(real_X, cycle_X)
    #### END CODE HERE ####
    return cycle_loss, cycle_X


def get_gen_loss(real_A: torch.Tensor,
                 real_B: torch.Tensor,
                 gen_AB: nn.Module,
                 gen_BA: nn.Module,
                 disc_A: nn.Module,
                 disc_B: nn.Module,
                 adv_criterion: nn.Module,
                 identity_criterion: nn.Module,
                 cycle_criterion: nn.Module,
                 lambda_identity: float = 0.1,
                 lambda_cycle: int = 10
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Return the loss of the generator given inputs.
    Parameters:
        real_A: the real images from pile A
        real_B: the real images from pile B
        gen_AB: the generator for class A to B; takes images and returns the images 
            transformed to class B
        gen_BA: the generator for class B to A; takes images and returns the images 
            transformed to class A
        disc_A: the discriminator for class A; takes images and returns real/fake class A
            prediction matrices
        disc_B: the discriminator for class B; takes images and returns real/fake class B
            prediction matrices
        adv_criterion: the adversarial loss function; takes the discriminator 
            predictions and the true labels and returns a adversarial 
            loss (which you aim to minimize)
        identity_criterion: the reconstruction loss function used for identity loss
            and cycle consistency loss; takes two sets of images and returns
            their pixel differences (which you aim to minimize)
        cycle_criterion: the cycle consistency loss function; takes the real images from X and
            those images put through a X->Y generator and then Y->X generator
            and returns the cycle consistency loss (which you aim to minimize).
            Note that in practice, cycle_criterion == identity_criterion == L1 loss
        lambda_identity: the weight of the identity loss
        lambda_cycle: the weight of the cycle-consistency loss
    '''
    # Hint 1: Make sure you include both directions - you can think of the generators as collaborating
    # Hint 2: Don't forget to use the lambdas for the identity loss and cycle loss!
    #### START CODE HERE ####
    # Adversarial Loss -- get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion)
    adv_loss_AB, _ = get_gen_adversarial_loss(
        real_A, disc_B, gen_AB, adv_criterion)
    adv_loss_BA, _ = get_gen_adversarial_loss(
        real_B, disc_A, gen_BA, adv_criterion)
    # Identity Loss -- get_identity_loss(real_X, gen_YX, identity_criterion)
    identity_loss_AB, _ = get_identity_loss(real_A, gen_BA, identity_criterion)
    identity_loss_BA, _ = get_identity_loss(real_B, gen_AB, identity_criterion)
    # Cycle-consistency Loss -- get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion)
    fake_A = gen_BA(real_B)
    fake_B = gen_AB(real_A)
    cycle_loss_AB, _ = get_cycle_consistency_loss(
        real_A, fake_B, gen_BA, cycle_criterion)
    cycle_loss_BA, _ = get_cycle_consistency_loss(
        real_B, fake_A, gen_AB, cycle_criterion)
    # Total loss
    gen_loss = adv_loss_AB + adv_loss_BA + lambda_identity * \
        (identity_loss_AB + identity_loss_BA) + \
        lambda_cycle*(cycle_loss_AB+cycle_loss_BA)
    #### END CODE HERE ####
    return gen_loss, fake_A, fake_B
