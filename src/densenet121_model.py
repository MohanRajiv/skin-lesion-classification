from torch import nn
from torchvision import transforms
from torchvision.models import DenseNet121_Weights, densenet121

NUM_CLASSES = 7

def build_densenet121(pretrained: bool = True, num_classes: int = NUM_CLASSES) -> nn.Module:
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

def densenet121_transforms(training: bool = False):
    evaluation_transform = DenseNet121_Weights.DEFAULT.transforms()

    if not training:
        return evaluation_transform
    
    return transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), evaluation_transform])