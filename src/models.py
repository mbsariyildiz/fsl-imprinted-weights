import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Net(nn.Module):

    def __init__(self, n_classes=100, d_emb=64, normalize=True, scale=True):
        super().__init__()
        self.n_classes = n_classes
        self.d_emb = d_emb
        self.normalize = normalize
        self.scale = scale

        self.extractor = Extractor()
        self.embedding = Embedding(d_emb, normalize)
        if n_classes > 0:
            self.classifier = Classifier(n_classes, d_emb, normalize)
        if scale:
            self.s = nn.Parameter(torch.FloatTensor([10]))

    def forward(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        x = self.classifier(x)
        if self.scale: x = self.s * x
        return x

    def extract(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        return x

    def extra_repr(self):
        # include model specific parameters
        extra_str = 'n_classes={n_classes}, ' \
                    'd_emb={d_emb}, ' \
                    'normalize={normalize}, ' \
                    'scale={scale}, '.format(**self.__dict__)
        return extra_str

class Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        basenet = models.resnet50(pretrained=True)
        self.extractor = nn.Sequential(*list(basenet.children())[:-1])

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        return x

class Embedding(nn.Module):

    def __init__(self, d_emb=64, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.d_emb = d_emb
        self.fc = nn.Linear(2048, d_emb)

    def forward(self, x):
        x = self.fc(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x

    def extra_repr(self):
        # include model specific parameters
        extra_str = 'd_emb={d_emb}, ' \
                    'normalize={normalize}, '.format(**self.__dict__)
        return extra_str

class Classifier(nn.Module):

    def __init__(self, n_classes, d_emb=64, normalize=True):
        super().__init__()
        self.n_classes = n_classes
        self.d_emb = d_emb
        self.normalize = normalize
        self.fc = nn.Linear(d_emb, n_classes, bias=False)

    def forward(self, x):
        if self.normalize:
            w = self.fc.weight
            w = F.normalize(w, dim=1, p=2)
            x = F.linear(x, w)
        else: 
            x = self.fc(x)
        return x

    def extra_repr(self):
        # include model specific parameters
        extra_str = 'n_classes={n_classes}, ' \
                    'd_emb={d_emb}, ' \
                    'normalize={normalize}, '.format(**self.__dict__)
        return extra_str
