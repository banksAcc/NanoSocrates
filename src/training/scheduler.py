import math

def cosine_with_warmup(step, warmup, max_steps):
    if step < warmup: return step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
