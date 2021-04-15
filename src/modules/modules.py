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
        self.student_model.train(False)
        return

    def forward(self, input):
        output = {}
        true_output = input['target']
        teacher_output = self.teacher_model({'data': input['data']})['target']
        supervise_loss = F.cross_entropy(teacher_output, true_output)
        with torch.no_grad():
            student_output = self.student_model({'data': input['data']})['target']
        teach_loss = kld_loss(teacher_output, student_output.detach())
        output['loss'] = supervise_loss + teach_loss
        output['target'] = teacher_output
        return output


class Student(nn.Module):
    def __init__(self, teacher_model, student_model, student_threshold):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.teacher_model.train(False)
        self.student_model.train(True)
        self.student_threshold = student_threshold

    def train_(self, train):
        self.teacher_model.train(False)
        self.student_model.train(train)
        return

    def make_indicator(self, teacher_output):
        p = F.softmax(teacher_output, dim=-1)
        max_p, _ = torch.max(p, dim=-1)
        indicator = (max_p >= self.student_threshold)
        return indicator

    def forward(self, input):
        output = {}
        with torch.no_grad():
            teacher_output = self.teacher_model({'data': input['data']})['target']
            indicator = self.make_indicator(teacher_output)
        if torch.all(indicator == 0):
            output['loss'] = None
        else:
            student_output = self.student_model({'data': input['data']})['target']
            teach_loss = kld_loss(student_output[indicator], teacher_output[indicator].detach())
            uda_output = self.student_model({'data': input['uda'][indicator]})['target']
            uda_loss = kld_loss(uda_output, teacher_output[indicator].detach(), T=0.4)
            output['loss'] = teach_loss + uda_loss
            output['target'] = student_output
        return output
