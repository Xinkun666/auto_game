import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image


class _EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class GameSceneClassifier:
    CLASS_LABELS = {0: 'indoor', 1: 'outdoor', 2: 'rooftop'}

    def __init__(self, checkpoint_path, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.class_to_idx = {'indoor': 0, 'outdoor': 1, 'rooftop': 2}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.model = _EfficientNetClassifier(num_classes=3)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, img):
        if isinstance(img, str):
            image = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            image = Image.fromarray(img)
        elif isinstance(img, Image.Image):
            image = img.convert('RGB')
        else:
            raise ValueError("img must be a file path (str), numpy.ndarray, or PIL.Image")

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        predicted_idx = predicted.item()

        return predicted_idx


if __name__ == '__main__':
    classifier = GameSceneClassifier('checkpoints/best_model.pth')

    result = classifier.predict('path/to/image.jpg')
    print(f"Class: {result} ({GameSceneClassifier.CLASS_LABELS[result]})")
