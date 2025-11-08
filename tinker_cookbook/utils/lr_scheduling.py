import logging

logger = logging.getLogger(__name__)


def compute_schedule_lr_multiplier(lr_schedule: str, step: int, total_steps: int) -> float:
    """
    What factor to multiply the base LR by due to the LR schedule
    """
    if lr_schedule == "linear":
        return 1 - step / total_steps
    elif lr_schedule == "constant":
        return 1
    elif lr_schedule == "cosine":
        # Cosine annealing from 1.0 to 0.0 (or min_lr)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    else:
        raise ValueError(f"Unknown learning rate schedule: {lr_schedule}")
