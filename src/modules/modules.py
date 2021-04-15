import torch
import torch.nn as nn
from config import cfg
from models import kld_loss, cross_entropy_loss
import torch.nn.functional as F


class Center(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        output = {}
        true_output = input['target']
        center_output = self.model({'data': input['data']})['target']
        supervise_loss = F.cross_entropy(center_output, true_output)
        output['loss'] = supervise_loss
        output['target'] = center_output
        return output


class User(nn.Module):
    def __init__(self, model, threshold):
        super().__init__()
        self.model = model
        self.threshold = threshold

    def make_label(self, output):
        soft_pseudo_label = F.softmax(output, dim=-1)
        max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_p.ge(self.threshold)
        return hard_pseudo_label, mask

    def forward(self, input):
        output = {}
        user_output = self.model({'data': input['data']})['target']
        hard_pseudo_label, mask = self.make_label(user_output)
        if torch.all(~mask):
            output['loss'] = None
        else:
            print(mask.float().sum())
            user_loss = kld_loss(user_output[mask], user_output[mask].detach(), T=0.4)
            uda_output = self.model({'data': input['uda'][mask]})['target']
            uda_loss = kld_loss(uda_output, user_output[mask].detach())
            output['loss'] = user_loss + uda_loss
        output['target'] = user_output
        return output
