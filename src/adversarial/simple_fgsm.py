import torch

def fgsm_attack(signal, epsilon=0.01):
    noise = torch.randn_like(signal) * epsilon
    adv_signal = signal + noise
    return adv_signal