from torch import nn
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

NUM_CLASSES = 7

def build_efficientnet_b0(pretrained: bool = True, num_classes: int = NUM_CLASSES) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model

def efficientnet_b0_transforms(training: bool = False):
    evaluation_transform = EfficientNet_B0_Weights.DEFAULT.transforms()

    if not training:
        return evaluation_transform
    
    return transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), evaluation_transform])
