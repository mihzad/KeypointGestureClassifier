import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import math

def create_scheduler(optimizer: torch.optim.Optimizer, scheduler_params: dict):

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=cosannealing_decay_warmup(**scheduler_params)
    )
    #scheduler = CosineAnnealingLR(
    #    optimizer=optimizer,
    #    T_max=50,
    #    eta_min=1e-8,
    #)

    return scheduler

def cosannealing_decay_warmup(warmup_steps, T_0, T_mult, decay_factor, base_lr, eta_min):
    # returns the func that performs all the calculations.
    # useful for keeping all the params in one place = scheduler def.
    def lr_lambda(epoch): #0-based epoch
        if epoch < warmup_steps:
            return base_lr * ((epoch + 1) / warmup_steps)

        annealing_step = epoch - warmup_steps

        # calculating which cycle (zero-based) are we in,
        # current cycle length (T_current) and position inside the cycle (t)
        if T_mult == 1:
            cycle = annealing_step // T_0
            t = annealing_step % T_0
            T_current = T_0

        else:
            # fast log-based computation
            cycle = int(math.log((annealing_step * (T_mult - 1)) / T_0 + 1, T_mult))
            sum_steps_of_previous_cycles = T_0 * (T_mult ** cycle - 1) // (T_mult - 1)
            t = annealing_step - sum_steps_of_previous_cycles
            T_current = T_0 * (T_mult ** cycle)


        # enable decay
        eta_max = base_lr * (decay_factor ** cycle)

        # cosine schedule between (eta_min, max_lr]
        lr = eta_min + 0.5 * (eta_max-eta_min) * (1 + math.cos(math.pi * t / T_current))
        return lr/base_lr

    return lr_lambda