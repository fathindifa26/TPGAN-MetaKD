# models/EfficientNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class EfficientNetWrapper(nn.Module):
    """
    Wrapper untuk EfficientNet dari timm library
    Kompatibel dengan framework TP-GAN
    """
    def __init__(
        self,
        model_name='efficientnet_b0',
        num_classes=347,
        dropout=0.5,
        pretrained=True,
        **kwargs
    ):
        super(EfficientNetWrapper, self).__init__()
        
        # Load EfficientNet dari timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout
        )
        
        # Get feature dimension dari backbone
        if hasattr(self.backbone, 'num_features'):
            self.feature_dim = self.backbone.num_features
        else:
            # Fallback untuk beberapa model
            self.feature_dim = self.backbone.classifier.in_features
        
        # Replace classifier dengan custom one
        self.backbone.classifier = nn.Identity()
        
        # Custom classifier untuk output yang konsisten
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # Initialize classifier
        nn.init.normal_(self.classifier.weight, 0, 0.01)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x, use_dropout=False):
        # Extract features
        features = self.backbone(x)
        
        # Apply dropout if requested
        if use_dropout:
            features = self.dropout(features)
        
        # Classifier
        logits = self.classifier(features)
        
        return logits, features

def efficientnet_b0(**kwargs):
    """EfficientNet-B0: 5.3M parameters"""
    return EfficientNetWrapper(model_name='efficientnet_b0', **kwargs)

def efficientnet_b1(**kwargs):
    """EfficientNet-B1: 7.8M parameters"""
    return EfficientNetWrapper(model_name='efficientnet_b1', **kwargs)

def efficientnet_b2(**kwargs):
    """EfficientNet-B2: 9.2M parameters"""
    return EfficientNetWrapper(model_name='efficientnet_b2', **kwargs)

def efficientnet_b3(**kwargs):
    """EfficientNet-B3: 12M parameters"""
    return EfficientNetWrapper(model_name='efficientnet_b3', **kwargs)

def efficientnet_b4(**kwargs):
    """EfficientNet-B4: 19M parameters"""
    return EfficientNetWrapper(model_name='efficientnet_b4', **kwargs)

def efficientnet_b5(**kwargs):
    """EfficientNet-B5: 30M parameters"""
    return EfficientNetWrapper(model_name='efficientnet_b5', **kwargs)

def efficientnet_b6(**kwargs):
    """EfficientNet-B6: 43M parameters"""
    return EfficientNetWrapper(model_name='efficientnet_b6', **kwargs)

def efficientnet_b7(**kwargs):
    """EfficientNet-B7: 66M parameters"""
    return EfficientNetWrapper(model_name='efficientnet_b7', **kwargs)

# Alias untuk kompatibilitas
EfficientNet_B0 = efficientnet_b0
EfficientNet_B1 = efficientnet_b1
EfficientNet_B2 = efficientnet_b2
EfficientNet_B3 = efficientnet_b3
EfficientNet_B4 = efficientnet_b4
EfficientNet_B5 = efficientnet_b5
EfficientNet_B6 = efficientnet_b6
EfficientNet_B7 = efficientnet_b7 