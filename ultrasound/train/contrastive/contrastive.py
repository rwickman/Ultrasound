import torch
import torch.nn as nn
import torch.nn.functional as F

from ultrasound.train.config import *
from ultrasound.train.model import SignalReduce, SignalAtt


        
class ContrastiveHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(EMB_SIZE, EMB_SIZE),
            nn.ReLU(),
            nn.Linear(EMB_SIZE, EMB_SIZE)
        )

    def forward(self, x):
        x = self.head(x)
        return x

class ContrastiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sig_reduce = SignalReduce()
        self.sig_att = SignalAtt()
        self.proj_head = ContrastiveHead()

    
    def forward(self, x):
        x = self.sig_reduce(x)
        x = self.sig_att(x)
        x = self.proj_head(x.reshape(-1, x.shape[2]))

        return x


def info_nce_loss(features):

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)
    print(labels, labels.shape, labels[0], labels[1])

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    #logits = logits / self.args.temperature
    return logits, labels