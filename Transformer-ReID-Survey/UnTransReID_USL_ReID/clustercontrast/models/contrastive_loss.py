import torch
import torch.nn as nn


class ConLoss(object):
    """
    Contrastive loss.
    """

    def __init__(self, temperature=0.8):
        self.temperature = temperature

    def __call__(self, z1, z2):
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)

        logits = z1 @ z2.T
        logits /= self.temperature
        n = z2.shape[0]
        labels = torch.arange(0, n, dtype=torch.long).cuda()
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss