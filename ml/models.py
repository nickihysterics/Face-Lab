"""Модели для извлечения эмбеддингов и классификации с ArcFace."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetBackbone(nn.Module):
    """Backbone на ResNet-50 для получения эмбеддингов лиц."""

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        """Инициализирует сеть с заданной размерностью эмбеддинга."""
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        modules = list(base.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        self.embedding = nn.Linear(base.fc.in_features, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Вычисляет L2-нормированные эмбеддинги для батча изображений."""
        feats = self.feature_extractor(x)
        feats = feats.view(feats.size(0), -1)
        feats = self.embedding(feats)
        feats = self.bn(feats)
        return F.normalize(feats, p=2, dim=1)


class ArcMarginProduct(nn.Module):
    """Слой ArcFace для увеличения межклассового расстояния."""

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50):
        """Создаёт ArcFace-слой с масштабом и угловой маржой."""
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Возвращает логиты с угловой маржой для обучения."""
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        phi = torch.cos(theta + self.m)
        threshold = math.cos(math.pi - self.m)
        phi = torch.where(cosine > threshold, phi, cosine - math.sin(math.pi - self.m) * self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class FaceClassifier(nn.Module):
    """Классификатор лиц на основе эмбеддингов и ArcFace."""

    def __init__(self, num_classes: int, embedding_dim: int = 512, pretrained: bool = True):
        """Собирает backbone и ArcFace-голову для заданного числа классов."""
        super().__init__()
        self.backbone = ResNetBackbone(embedding_dim=embedding_dim, pretrained=pretrained)
        self.arc = ArcMarginProduct(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Прямой проход обучения с учетом меток."""
        emb = self.backbone(x)
        logits = self.arc(emb, label)
        return logits

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает эмбеддинги без классификационной головы."""
        return self.backbone(x)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Inference logits without ground-truth labels or the training margin."""
        embeddings = self.extract(x)
        return F.linear(F.normalize(embeddings), F.normalize(self.arc.weight)) * self.arc.s
