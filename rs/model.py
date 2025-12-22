import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

class PositionPredictor(nn.Module):
    def __init__(self, input_size=511, hidden_size=512, output_size=255, dropout=0.1):
        super().__init__()
        
        layers = []
        
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(3):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.2, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        return sigmoid_focal_loss(
            inputs, targets, 
            alpha=self.alpha, 
            gamma=self.gamma, 
            reduction='mean'
        )
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)