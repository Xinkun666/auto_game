import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image


class _EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=4, weights=None):
        super().__init__()
        # 场景分类器随后会完整加载本地 scene_best_model.pth，禁止 torchvision
        # 在启动自动化时联网下载 ImageNet 预训练权重。
        self.backbone = models.efficientnet_b0(weights=weights)
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
    CLASS_LABELS = {0: 'indoor', 1: 'outdoor', 2: 'nearwall', 3: 'nearhouse'}

    def __init__(self, checkpoint_path, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.class_to_idx = {name: idx for idx, name in self.CLASS_LABELS.items()}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.model = _EfficientNetClassifier(num_classes=len(self.CLASS_LABELS))
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if checkpoint.get('class_to_idx') not in (None, self.class_to_idx):
            raise ValueError(f"模型类别不匹配：{checkpoint['class_to_idx']} != {self.class_to_idx}")
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
            predicted = outputs.argmax(dim=1)

        return self.idx_to_class[predicted.item()]


if __name__ == '__main__':
    import sys

    result = GameSceneClassifier(sys.argv[1]).predict(sys.argv[2])
    assert result in GameSceneClassifier.CLASS_LABELS.values()
    print(result)
