import torch
from torch.optim import lr_scheduler

def get_lr_scheduler(optimizer, config: dict):
    """
    Returns the learning rate scheduler based on the configuration.

    Args:
    - optimizer: The optimizer to which the scheduler will be applied.
    - config (dict): Configuration dictionary containing the scheduler type and settings.
    
    Returns:
    - A learning rate scheduler object.
    """
    lr_schedule = config.get("lr_schedule", "linear")  # Default to 'linear'
    lr_settings = config.get("lr_settings", {})

    if lr_schedule == "linear":
        # Linear decay scheduler (lr will decay from initial value to 0)
        total_iters = lr_settings.get("total_iters", 1000)  # Total iterations to decay over
        return lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_iters)

    elif lr_schedule == "cosine":
        # Cosine Annealing scheduler (lr follows a cosine decay curve)
        T_max = lr_settings.get("T_max", 50)  # Max number of iterations for full annealing
        eta_min = lr_settings.get("eta_min", 0)  # Minimum learning rate after annealing
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    elif lr_schedule == "step":
        # StepLR scheduler (lr decays every step by a given factor)
        step_size = lr_settings.get("step_size", 30)  # Number of steps before decaying
        gamma = lr_settings.get("gamma", 0.1)  # Decay factor
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule == "exponential":
        # ExponentialLR scheduler (lr decays exponentially)
        gamma = lr_settings.get("gamma", 0.99)  # Decay factor
        return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    elif lr_schedule == "reduce_on_plateau":
        # ReduceLROnPlateau scheduler (lr is reduced when a metric plateaus)
        mode = lr_settings.get("mode", "min")  # Mode to monitor (min/max)
        factor = lr_settings.get("factor", 0.1)  # Factor by which the learning rate will be reduced
        patience = lr_settings.get("patience", 10)  # Number of epochs with no improvement before reducing
        threshold = lr_settings.get("threshold", 1e-4)  # Threshold for measuring the new optimum
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                                               threshold=threshold)

    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")