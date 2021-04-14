import torch
import torch.nn as nn
from config import cfg
from models import kld_loss
import torch.nn.functional as F


class Teacher(nn.Module):
    def __init__(self, teacher_model, student_model):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.teacher_model.train(True)
        self.student_model.train(False)

    def train_(self, train):
        self.teacher_model.train(train)
        return

    def forward(self, input):
        output = {}
        true_output = input['target']
        teacher_output = self.teacher_model({'data': input['data']})['target']
        with torch.no_grad():
            student_output = self.student_model({'data': input['data']})['target']
        supervise_loss = F.cross_entropy(teacher_output, true_output)
        teach_loss = kld_loss(teacher_output, student_output.detach())
        output['loss'] = supervise_loss + teach_loss
        output['target'] = teacher_output
        return output


class Student(nn.Module):
    def __init__(self, teacher_model, student_model):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.teacher_model.train(False)
        self.student_model.train(True)

    def train_(self, train):
        self.student_model.train(train)
        return

    def forward(self, input):
        output = {}
        student_output = self.student_model({'data': input['data']})['target']
        uda_output = self.student_model({'data': input['uda']})['target']
        with torch.no_grad():
            teacher_output = self.teacher_model({'data': input['data']})['target']
        teach_loss = kld_loss(student_output, teacher_output.detach())
        uda_loss = kld_loss(uda_output, student_output.detach())
        output['loss'] = teach_loss + uda_loss
        output['target'] = student_output
        return output
