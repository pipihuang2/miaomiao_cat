import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification, AutoImageProcessor
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from tqdm import tqdm

image_cache = {}

class QRCodeDataset(Dataset):
    """
    假设有一个文件夹 images_dir, 里面有若干拍摄/合成后的失真二维码图
    以及一个 label_dict，结构大致为:
        {
            "img_0001.jpg": [0,1,1,0,...(长度196)],
            "img_0002.jpg": [1,0,1,0,...(长度196)],
            ...
        }
    """
    def __init__(self, images_dir, label_dict, transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.label_dict = label_dict
        self.filenames = list(self.label_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        if fname in image_cache:
            image = image_cache[fname]
        else:
            path = os.path.join(self.images_dir, fname)
            image = Image.open(path).convert("L")
            image_cache[fname] = image

        if self.transform:
            image = self.transform(image)

        # label: [0/1,...(196)]
        label = self.label_dict[fname]
        label_tensor = torch.tensor(label, dtype=torch.float32)  # shape: [196]

        return image, label_tensor


# ============ 2. 创建模型 (transformers) ============

def create_vit_model():
    """
    创建一个 ViTForImageClassification 实例:
      - num_labels=196, 对应二维码 14x14
      - problem_type="multi_label_classification", 用 BCE
    """

    model = ViTForImageClassification.from_pretrained("WinKawaks/vit-tiny-patch16-224",
        num_labels=196,
        ignore_mismatched_sizes=True,
        num_channels=1,
        problem_type="multi_label_classification")
    return model



def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct_count = 0
    total_count = 0

    # 使用 tqdm 包装 dataloader
    progress_bar = tqdm(dataloader, desc="Training", leave=True)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)  # shape: [B,196]

        # forward
        outputs = model(images, labels=labels)  
        # 当 problem_type="multi_label_classification" 时,
        # outputs.loss 采用 BCEWithLogitsLoss

        loss = outputs.loss  
        logits = outputs.logits  # shape: [B,196]

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = (torch.sigmoid(logits) > 0.5).float()  # [B,196]
        correct_count += (preds == labels).float().sum().item()
        total_count   += labels.numel()

        # 计算当前 accuracy
        accuracy = correct_count / total_count

        # 更新进度条描述信息
        # progress_bar.set_postfix(loss=total_loss / total_count, accuracy=accuracy)
        progress_bar.set_postfix(
            loss=f"{total_loss / total_count:.9f}", 
            acc=f"{accuracy:.9f}",
            diff=f"{total_count - correct_count}",
            lr=optimizer.param_groups[0]['lr']
        )
    accuracy = correct_count / total_count
    return total_loss / len(dataloader.dataset), accuracy

def eval_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct_count = 0
    total_count = 0

    # 这里要注意: ViTForImageClassification 的 forward(labels=labels) 才会返回 loss
    # 如果只是推理,可以不传 labels, 然后手动计算 loss
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)       # forward pass without labels
            logits = outputs.logits       # [B,196]
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            # 二值化预测
            preds = (torch.sigmoid(logits) > 0.5).float()  # [B,196]

            correct_count += (preds == labels).float().sum().item()
            total_count   += labels.numel()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct_count / total_count
    return avg_loss, accuracy


# ============ 4. 主流程 (示例) ============

def generate_01_matrix(content):
    from pylibdmtx.pylibdmtx import encode
    import numpy as np
    import matplotlib.pyplot as plt
    encoded = encode(content.encode('utf8'), size='14x14')
    _bytes = encoded.pixels
    img = np.frombuffer(_bytes, dtype=np.uint8).reshape(encoded.height, encoded.width, 3)
    matrix = (img[10:-10:5, 10:-10:5, 0] // 255)
    # plt.imshow(matrix, cmap='gray')
    # plt.show()
    matrix = matrix.reshape(-1).tolist()
    return matrix

import os
images_dir = r"D:\data_backup\all"

train_dir = os.path.join(images_dir, "train")
test_dir = os.path.join(images_dir, "test")
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)


import os
def build_label_dict(images_dir):
    images = os.listdir(images_dir)
    result = {}
    for name in images:
        prefix = name.split('.jpg')[0]
        if len(prefix) == 11:
            result[name] = generate_01_matrix(prefix)
    return result

label_dict = build_label_dict(train_dir)
test_label_dict = build_label_dict(test_dir)

import random
import torchvision.transforms.functional as F

class RandomFixedRotation:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return F.rotate(img, angle)

class RandomCropAroundEdges:
    def __init__(self, max_crop=15):
        self.max_crop = max_crop

    def __call__(self, img):
        width, height = img.size
        left = random.randint(0, self.max_crop)
        top = random.randint(0, self.max_crop)
        right = random.randint(0, self.max_crop)
        bottom = random.randint(0, self.max_crop)
        return img.crop((left, top, width - right, height - bottom))

transform = transforms.Compose([
    RandomCropAroundEdges(max_crop=15),
    transforms.ColorJitter(brightness=0.5),  
    transforms.RandomRotation(degrees=(0, 15)), 
    transforms.Resize((112, 112)),
    RandomFixedRotation([0, 90, 180, 270]),
    transforms.ToTensor(),
])

dataset = QRCodeDataset(train_dir, label_dict, transform=transform)
testdataset = QRCodeDataset(test_dir, test_label_dict, transform=transform)


dataloader = DataLoader(dataset, batch_size=96, shuffle=True)
testdataloader = DataLoader(testdataset, batch_size=96, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = create_vit_model().to(device)

# model_path = "tiny-vit-dm-196-120K-aug.v3"
# model = ViTForImageClassification.from_pretrained(model_path)
# model = model.to(device)

from transformers import ViTConfig
# config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
config = ViTConfig.from_pretrained("WinKawaks/vit-tiny-patch16-224")
config.patch_size = 8
config.image_size = 112
config.hidden_size = 192
config.num_hidden_layers = 8
config.num_attention_heads = 6
config.intermediate_size = 394
config.problem_type = "multi_label_classification"
config.num_labels = 196
config.num_channels = 1
model = ViTForImageClassification(config=config).to(device)

# model_path = "vit-base"
# model = ViTForImageClassification.from_pretrained(model_path).model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
transform = transforms.Compose([
    RandomCropAroundEdges(max_crop=15),
    transforms.ColorJitter(brightness=0.5),  
    transforms.RandomRotation(degrees=(-10, 10)), 
    transforms.Resize((112, 112)),
    RandomFixedRotation([0, 90, 180, 270]),
    transforms.ToTensor(),
])

dataloader.dataset.transform = transform
testdataloader.dataset.transform = transform

from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# from torch.optim.lr_scheduler import StepLR
# scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

epochs = 1000
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, dataloader, optimizer, device)
    model.save_pretrained("supertiny2-vit")
    scheduler.step(train_loss)
    if epoch % 10 == 9:
        val_loss, val_acc = eval_one_epoch(model, testdataloader, device)
        print(f"Epoch [{epoch+1}/{epochs}]  "
                f"Train Loss: {train_loss:.9f}  "
                f"Train Acc: {train_acc:.9f}  "
                f"Val Loss: {val_loss:.9f}  "
                f"Val Acc: {val_acc:.9f} "
                f"LR: {optimizer.param_groups[0]['lr']:.9f}")
    # scheduler.step()

