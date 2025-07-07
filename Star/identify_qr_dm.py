import glob

import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
from pylibdmtx import pylibdmtx
import cv2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_01_matrix(content):
    from pylibdmtx.pylibdmtx import encode
    import numpy as np
    import matplotlib.pyplot as plt
    encoded = encode(content.encode('utf8'), size='14x14')
    _bytes = encoded.pixels
    img = np.frombuffer(_bytes, dtype=np.uint8).reshape(encoded.height, encoded.width, 3)
    matrix = (img[10:-10:5, 10:-10:5, 0] // 255)
    matrix = matrix.reshape(-1).tolist()
    return matrix


class ViTDataMatrixDecoder:
    def __init__(self, model_name):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained(model_name).to(device)

        self._transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def decode(self, image):
        image = Image.fromarray(image.astype(np.uint8)).convert('RGB')
        image = self._transforms(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image.to(device))

        logits = outputs.logits
        sigmoid_logits = torch.sigmoid(logits).float().cpu().numpy()
        conf = np.sum(sigmoid_logits > 0.9) + np.sum(sigmoid_logits < 0.1)

        preds = (torch.sigmoid(logits) > 0.5).float().reshape(-1, 14, 14).cpu()

        image = np.ones((18, 18)) * 255
        image[2:-2, 2:-2] = preds[0].numpy() * 255
        image = cv2.resize(image, (image.shape[1] * 5, image.shape[0] * 5), interpolation=cv2.INTER_NEAREST)

        decode = pylibdmtx.decode(image)
        if decode:
            data = decode[0].data.decode()
            base = generate_01_matrix(data)
            diff = np.sum(np.array(base) != preds[0].reshape(-1).numpy())
        else:
            diff = 0
            data = None
        return data, (image, diff, conf)

if __name__ == '__main__':
    import pathlib
    pic = pathlib.Path("F:\qr\DAB").rglob("*.bmp")
    # pic = glob.glob(r"F:\qr\ng_qr\*.jpg")
    dm_decoder = ViTDataMatrixDecoder(r'D:\Project\HYJ\cosmos\models/vit-dm')
    import shutil
    for pic_ in tqdm.tqdm(pic):
        image = cv2.imread(pic_)
        data, (_image, _diff, _conf) = dm_decoder.decode(image)
        print(f"data原始内容：{repr(data)}")
        if data and len(data) == 11:
            try:
                shutil.move(pic_,f"F:\qr\DAB_PASS\pass/{data}.jpg")
            except Exception as e:
                print(f'出现问题{e}')