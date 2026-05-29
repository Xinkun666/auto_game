import os
import cv2
import random
from glob import glob
from PIL import Image
from PIL import ImageEnhance

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, random_split

class CustomAugment:
    def __init__(self):
        self.rotation_range = 10  # ±10度
        self.flip_prob = 0.5
        self.color_jitter_prob = 0.5
        self.noise_prob = 0.3

    def __call__(self, img):
        # 随机水平翻转
        if random.random() < self.flip_prob:
            img = TF.hflip(img)

        # 随机旋转
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        img = TF.rotate(img, angle, fill=(0,))

        # 随机颜色扰动（亮度、对比度）
        if random.random() < self.color_jitter_prob:
            factor_b = random.uniform(0.7, 1.3)
            factor_c = random.uniform(0.7, 1.3)
            img = ImageEnhance.Brightness(img).enhance(factor_b)
            img = ImageEnhance.Contrast(img).enhance(factor_c)

        # 随机加噪
        if random.random() < self.noise_prob:
            img = self.add_noise(img)

        return img

    def add_noise(self, img):
        arr = torch.tensor(TF.to_tensor(img))
        noise = torch.randn_like(arr) * 0.05
        arr = torch.clamp(arr + noise, 0, 1)
        return TF.to_pil_image(arr)

class SpeedNumberDataset(Dataset):
    def __init__(self, root_dir, label_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        with open(os.path.join(root_dir, label_file), "r") as f:
            for line in f:
                path, label = line.strip().split()
                self.samples.append((path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        if int(label) == -1:
            label = 10
        img = Image.open(os.path.join(self.root_dir, img_path)).convert("L")

        if self.transform:
            img = self.transform(img)

        return img, label

class TinyDigitCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 输入通道1 -> 输出16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 5x10 -> 2x5

            # 第二层卷积
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 第三层卷积
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),

            # 全连接层
            nn.Linear(64 * 7 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_model(model, train_loader, val_loader, device, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)
    best_val_acc = 0.0  # 记录最好的验证集准确率

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss:.3f} | Train Acc: {acc:.3f}")

        # 验证
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                val_correct += (preds == labels).sum().item()

        val_acc = val_correct / len(val_loader.dataset)
        print(f"Validation Acc: {val_acc:.3f}\n")

        # 如果当前验证集准确率更好，则保存最佳模型权重
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "weights/speed_extent_model.pth")
            print(f"New best model saved with val_acc: {best_val_acc:.3f}\n")

    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r'aw/autogame/customs_examples/Auto_PUBG_ALL/resource/weights/speed_extent_model.pth'
model = TinyDigitCNN(num_classes=6)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

def get_speed_extent(frame_cv):

    img_pil = Image.fromarray(frame_cv).convert("L")

    transform_infer = transforms.Compose([
        transforms.Resize((15, 10)),               # 与训练一致
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    img_tensor = transform_infer(img_pil).unsqueeze(0).to(device)  # 增加 batch 维度

    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()

    return pred


def train_speed_number_model(
    root_dir: str,
    label_file: str = "label.txt",
    model_class=None,
    num_classes: int = 6,
    img_size=(15, 10),
    batch_size: int = 32,
    epochs: int = 50,
    save_path: str = "speed_extent_model.pth",
    augment: bool = True,
):
    """
    训练和平精英数字识别模型（带归一化与数据增强）

    参数:
        root_dir (str): 数据集根目录
        label_file (str): 标签文件名
        model_class (torch.nn.Module): 模型类，例如 TinyDigitCNN
        num_classes (int): 分类类别数
        img_size (tuple): 图像缩放尺寸
        batch_size (int): 批量大小
        epochs (int): 训练轮数
        save_path (str): 模型保存路径
        augment (bool): 是否启用数据增强
    """

    # ========== 1. 定义数据增强和标准化 ==========
    transform_train = transforms.Compose([
        CustomAugment() if augment else transforms.Lambda(lambda x: x),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # ========== 2. 构建完整数据集并划分 ==========
    full_dataset = SpeedNumberDataset(root_dir, label_file, transform=None)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    print(f"train_size: {train_size}, val_size: {val_size}")

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset.transform = transform_train
    val_dataset.dataset.transform = transform_val

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # ========== 3. 初始化模型与设备 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(num_classes=num_classes).to(device)

    # ========== 4. 执行训练 ==========
    trained_model = train_model(model, train_loader, val_loader, device, epochs=epochs)

    # ========== 5. 保存模型 ==========
    torch.save(trained_model.state_dict(), save_path)
    print(f"✅ 模型训练完成并保存至: {save_path}")

if __name__ == "__main__":

    ROI = (119, 347, 134, 357)

    temp_dir = r'D:\Resource\File\temp'
    visual_dir = r'D:\Resource\File\temp\visual'
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)

    png_files = glob(os.path.join(temp_dir, '*.jpg'))
    for png_file in png_files:
        frame_cv = cv2.imread(png_file)
        h,w = frame_cv.shape[:2]
        if h > w:
            frame_cv = cv2.rotate(frame_cv, cv2.ROTATE_90_CLOCKWISE)
        frame_cv_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        frame_cv_roi = frame_cv_rgb[ROI[1]:ROI[3], ROI[0]:ROI[2]]
        pred = get_speed_extent(frame_cv_roi)
        speed_extent = 'Empty'
        if pred == 0:
            speed_extent = 'Low_Speed'
        elif pred == 1:
            speed_extent = 'Mid_Low_Speed'
        elif pred == 2:
            speed_extent = 'Mid_High_Speed'
        elif pred == 3:
            speed_extent = 'High_Speed'
        elif pred == 4:
            speed_extent = 'Super_High_Speed'

        cv2.putText(frame_cv, speed_extent, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.rectangle(frame_cv, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (0,0,255), 2)
        cv2.imwrite(os.path.join(visual_dir, os.path.basename(png_file)), frame_cv)




    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_path = r'/utils/weights/speed_extent_model.pth'
    # model = TinyDigitCNN()
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model = model.to(device)
    # model.eval()
    #
    #
    # label_dir = r"D:\project\Python\auto_pubg\resource\validation\label.txt"
    # image_dir = r"D:\project\Python\auto_pubg\resource\validation"
    #
    # with open(label_dir, "r") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         line = line.strip()
    #         image_name, label = line.split()
    #         image_path = os.path.join(image_dir, image_name)
    #         pred, conf = predict(model, image_path, device)
    #         print(f"{image_path} -> {label} -> {pred} -> {conf:.2f}")

    # root_dir = r'D:\project\Python\auto_pubg_resource\spped_number_extent_val'
    # visual_dir = r'D:\project\Python\auto_pubg_resource\spped_number_extent_val\visual'
    # os.makedirs(visual_dir, exist_ok=True)
    #
    # jpg_files = glob(os.path.join(root_dir, "*.jpg"))
    # roi = (119, 347, 134, 357)
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_path = r'D:\project\Python\auto_pubg\utils\weights\speed_extent_model.pth'
    # model = TinyDigitCNN(num_classes=6)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model = model.to(device)
    # model.eval()
    #
    # for jpg_file in jpg_files:
    #     img = cv2.imread(jpg_file)
    #     if img is None:
    #         print(f"⚠️ Failed to read image: {jpg_file}")
    #         continue
    #
    #     # 如果图片竖直，则旋转90度并且**重新获取尺寸**
    #     height, width = img.shape[:2]
    #     if height > width:
    #         img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    #     # ------------ 在旋转之后马上重新读尺寸 ------------
    #     height, width = img.shape[:2]
    #
    #     # 检查 ROI 是否在图像范围内并裁剪（并处理越界）
    #     x1, y1, x2, y2 = roi
    #     # clamp roi 到图像边界
    #     x1 = max(0, min(x1, width - 1))
    #     x2 = max(0, min(x2, width))
    #     y1 = max(0, min(y1, height - 1))
    #     y2 = max(0, min(y2, height))
    #
    #     if x2 <= x1 or y2 <= y1:
    #         print(f"⚠️ ROI is invalid for {jpg_file}: {x1, y1, x2, y2} image size {width, height}")
    #         roi_img = None
    #     else:
    #         roi_img = img[y1:y2, x1:x2]
    #
    #     # 计算 speed（如果 roi_img 无效，跳过或设为特殊值）
    #     try:
    #         if roi_img is None or roi_img.size == 0:
    #             speed = "N/A"
    #         else:
    #             speed_val = get_speed(roi_img)  # 你的函数
    #             # 明确转换，避免 numpy 类型问题
    #             speed = str(speed_val)
    #     except Exception as e:
    #         print(f"⚠️ get_speed error for {jpg_file}: {e}")
    #         speed = "ERR"
    #
    #     # 确保是3通道 uint8 图（防止外部处理改成 float 或单通道）
    #     if img.dtype != "uint8":
    #         img = (img * 255).astype("uint8") if img.max() <= 1.0 else img.astype("uint8")
    #     if len(img.shape) == 2:
    #         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #
    #     # ========== 绘制 speed 在中下或左下（这里示例左下） ==========
    #     text = f"{speed}"
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #
    #     # 自适应字体和厚度（根据图片大小）
    #     base = max(width, height)
    #     font_scale = max(0.5, base / 800)  # 约 800 为参考分辨率，按需调整
    #     thickness = max(1, int(base / 400))
    #
    #     # 计算文字尺寸并确保坐标在图像内
    #     text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    #     pad = 8
    #     text_x = 10  # 左侧 10 px
    #     text_y = height - 10  # 距底部 10 px
    #
    #     # 如果文字超出宽度，则左移到能显示的位置
    #     if text_x + text_size[0] + pad * 2 > width:
    #         text_x = max(0, width - text_size[0] - pad * 2)
    #
    #     # 画半透明背景矩形，提升可见度
    #     overlay = img.copy()
    #     rect_tl = (text_x - pad, text_y - text_size[1] - pad)
    #     rect_br = (text_x + text_size[0] + pad, text_y + pad // 2)
    #     cv2.rectangle(overlay, rect_tl, rect_br, (0, 0, 0), -1)
    #     alpha = 0.5
    #     img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    #
    #     # 绘制文字（红色 BGR=(0,0,255)）
    #     color = (0, 0, 255)
    #     cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    #
    #     # 保存结果
    #     save_path = os.path.join(visual_dir, "annotated_" + os.path.basename(jpg_file))
    #     ok = cv2.imwrite(save_path, img)
    #     if not ok:
    #         print(f"❌ Failed to save: {save_path}")
    #     else:
    #         print(f"✅ Saved: {save_path}")

