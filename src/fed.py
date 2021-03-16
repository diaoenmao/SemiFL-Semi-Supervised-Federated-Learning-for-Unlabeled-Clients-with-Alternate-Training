import copy
import torch
import numpy as np
from config import cfg
from collections import OrderedDict


class Federation:
    def __init__(self, teacher_model, student_model, global_optimizer, data_split, target_split):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.global_optimizer = global_optimizer
        self.local_weight = self.make_local_weight
        self.target_split = target_split

    def make_local_weight(self):

        return

    def distribute(self, user_idx):
        student_parameters = self.student_model.state_dict()
        local_parameters = [copy.deepcopy(student_parameters) for _ in range(len(user_idx))]
        return local_parameters

    def update(self, local_parameters):
        self.global_optimizer.zero_grad()
        student_parameters = self.student_model.state_dict()
        for k, v in student_parameters.items():
            parameter_type = k.split('.')[-1]
            tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
            for m in range(len(local_parameters)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    tmp_v += self.local_weight[m] * local_parameters[m][k]
            v.grad = v - tmp_v.to(v.dtype)
        self.global_optimizer.step()
        student_parameters = self.student_model.state_dict()
        return student_parameters
