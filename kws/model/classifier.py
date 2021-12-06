import torch
import torch.nn as nn


class GlobalClassifier(nn.Module):
    """Add a global average pooling before the classifier"""
    def __init__(self, classifier: nn.Module):
        super(GlobalClassifier, self).__init__()
        self.classifier = classifier

    def forward(self, x: torch.Tensor):
        x = torch.mean(x, dim=1)
        return self.classifier(x)


class LastClassifier(nn.Module):
    """Select last frame to do the classification"""
    def __init__(self, classifier: nn.Module):
        super(LastClassifier, self).__init__()
        self.classifier = classifier

    def forward(self, x: torch.Tensor):
        x = x[:, -1, :]
        return self.classifier(x)

class ElementClassifier(nn.Module):
    """Classify all the frames in an utterance"""
    def __init__(self, classifier: nn.Module):
        super(ElementClassifier, self).__init__()
        self.classifier = classifier

    def forward(self, x: torch.Tensor):
        return self.classifier(x)
