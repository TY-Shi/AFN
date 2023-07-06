import torch
import torch.nn as nn


class ASCLoss(nn.Module):
    def __init__(self):
        super(ASCLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        # predcit_affinity NCHW,target_affinity NCHW
        assert predict.size() == target.size()  # "the size of predict and target must be equal."
        mult = predict * target
        mult_sumC = torch.sum(mult, dim=1)

        predict_mag = torch.norm(predict, dim=1)

        target_mag = torch.norm(target, dim=1)
        mult_mag = target_mag * predict_mag

        final_loss = 1 - torch.mean(mult_sumC / (mult_mag + self.epsilon))
        average_loss = final_loss

        return average_loss
