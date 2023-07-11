import copy
import torch
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class PTMOONLoss(_Loss):
    def __init__(self, mu, tau=0.5) -> None:
        """Compute MOON loss

        Args:
            mu: weighting parameter
            tau: temperature parameter
        """
        super().__init__()
        if mu < 0.0:
            raise ValueError("mu should be no less than 0.0")
        self.mu = mu
        self.tau = tau

    def forward(self, rep_local, reps_local_old, rep_global):
        sim_pos = F.cosine_similarity(rep_local, rep_global)
        logits = sim_pos.reshape(-1, 1)
        # TODO: use reps_old with decaying weights
        # Implemented without weights here
        for rep_local_old in reps_local_old:
            sim_neg = F.cosine_similarity(rep_local, rep_local_old)
            logits = torch.cat((logits, sim_neg.reshape(-1, 1)), dim=1)
        # Note that logits is a matrix with 2 columns. 
        # The first column is the positive pair and the second column is the negative pair.
        # Then, by setting the label to 0 and using cross-entropy loss, we can compute the contrastive loss.
        logits /= self.tau
        dummy_labels = torch.zeros(x.size(0)).to(logits.device()).long()
        contrast_loss = self.mu * torch.nn.CrossEntropyLoss(logits, dummy_labels)

        return contrast_loss

