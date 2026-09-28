import os
import json
import shutil
import random
from pathlib import Path
from collections import Counter


def parse_labelme_shapes(shapes, img_w, img_h):
    """
    将 LabelMe shapes 解析为归一化的 YOLO bbox：
    [class_id, x_center, y_center, width, height]
    目前支持 rectangle / polygon，统一用外接矩形转 bbox。
    """
    yolo_lines = []
    for shape in shapes:
        pts = shape.get("points", [])
        if len(pts) < 2:
            continue

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # 裁剪到图像边界内
        x_min = max(0.0, min(x_min, img_w))
        x_max = max(0.0, min(x_max, img_w))
        y_min = max(0.0, min(y_min, img_h))
        y_max = max(0.0, min(y_max, img_h))

        w = x_max - x_min
        h = y_max - y_min
        if w <= 0 or h <= 0:
            continue

        x_center = x_min + w / 2.0
        y_center = y_min + h / 2.0

        # 归一化
        yolo_lines.append([
            x_center / img_w,
            y_center / img_h,
            w / img_w,
            h / img_h,
        ])

    return yolo_lines


def find_image_file(json_dir, image_path_hint):
    """
    根据 JSON 中的 imagePath，在 json_dir 下查找实际图片文件。
    支持 .jpg .jpeg .png .bmp .tif .tiff .webp
    """
    if not image_path_hint:
        return None

    # 先按原文件名试
    candidate = os.path.join(json_dir, image_path_hint)
    if os.path.isfile(candidate):
        return candidate

    base = os.path.splitext(os.path.basename(image_path_hint))[0]
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

    for ext in exts:
        candidate = os.path.join(json_dir, base + ext)
        if os.path.isfile(candidate):
            return candidate

    # 大小写不敏感再试一次
    for ext in exts:
        for variant in {ext, ext.upper()}:
            candidate = os.path.join(json_dir, base + variant)
            if os.path.isfile(candidate):
                return candidate

    return None


def convert_one(json_path, class_names, label_mapping=None, allow_new_classes=False):
    """
    转换单个 JSON 文件，返回 (image_src_path, yolo_lines, img_w, img_h, class_names)。
    如果 class_names 为 None，则自动收集类别并返回更新后的 class_names。
    allow_new_classes: 为 True 时，遇到新类别会自动加入 class_names（用于收集阶段）。
    """
    json_dir = os.path.dirname(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data.get("imageWidth")
    img_h = data.get("imageHeight")
    image_path_hint = data.get("imagePath", "")

    # 如果没有高宽，尝试从图片读取
    if not img_w or not img_h:
        img_path = find_image_file(json_dir, image_path_hint)
        if img_path:
            try:
                from PIL import Image
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception:
                pass

    if not img_w or not img_h:
        raise ValueError(f"无法获取图片尺寸: {json_path}")

    img_path = find_image_file(json_dir, image_path_hint)
    if not img_path:
        raise FileNotFoundError(f"未找到图片: {image_path_hint} (JSON: {json_path})")

    auto_classes = class_names is None
    if auto_classes:
        class_names = []

    yolo_lines = []
    for shape in data.get("shapes", []):
        raw_label = shape.get("label", "unknown")
        label = raw_label
        if label_mapping and raw_label in label_mapping:
            label = label_mapping[raw_label]

        if label not in class_names:
            if auto_classes or allow_new_classes:
                class_names.append(label)
            else:
                # 跳过未知类别，避免训练时报错
                print(f"  警告：未知类别 '{label}'，已跳过。JSON: {json_path}")
                continue

        class_id = class_names.index(label)
        boxes = parse_labelme_shapes([shape], img_w, img_h)
        for box in boxes:
            yolo_lines.append([class_id] + box)

    return img_path, yolo_lines, img_w, img_h, class_names


def split_dataset(json_files, val_ratio=0.2, seed=42):
    """按文件级别随机划分 train/val。"""
    random.seed(seed)
    files = json_files.copy()
    random.shuffle(files)

    val_num = max(1, int(len(files) * val_ratio)) if len(files) > 1 else 0
    val_files = files[:val_num]
    train_files = files[val_num:]
    return train_files, val_files


def build_dataset(
    input_dir,
    output_dir,
    val_ratio=0.2,
    class_names=None,
    label_mapping=None,
    seed=42,
):
    """
    将 input_dir 下的 LabelMe JSON + 图片转换为 YOLO 格式。

    参数：
        input_dir:    输入目录，需包含 .json 和图片文件
        output_dir:   YOLO 输出目录
        val_ratio:    验证集比例（默认 0.2）
        class_names:  类别名称列表。为 None 时自动从 JSON 中收集
        label_mapping: 标签映射字典，例如 {"断胶": "ng", "气泡": "ng"}
        seed:         随机种子
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"未在 {input_dir} 下找到 JSON 文件")
        return

    print(f"共找到 {len(json_files)} 个 JSON 文件")

    # 创建目录
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    # 第一次遍历：自动收集类别（如果需要）
    if class_names is None:
        class_names = []
        print("正在自动收集类别...")
        for jp in json_files:
            try:
                _, _, _, _, class_names = convert_one(
                    str(jp),
                    class_names=class_names,
                    label_mapping=label_mapping,
                    allow_new_classes=True,
                )
            except Exception as e:
                print(f"  收集类别时出错: {jp.name}, {e}")
        print(f"自动收集到类别: {class_names}")

    # 划分 train / val
    train_files, val_files = split_dataset([str(p) for p in json_files], val_ratio, seed)
    print(f"训练集: {len(train_files)}，验证集: {len(val_files)}")

    splits = {
        "train": train_files,
        "val": val_files,
    }

    stats = Counter()

    for split_name, files in splits.items():
        for json_path in files:
            try:
                img_path, yolo_lines, _, _, _ = convert_one(
                    json_path, class_names=class_names, label_mapping=label_mapping
                )
            except Exception as e:
                print(f"  转换失败: {json_path}, 原因: {e}")
                continue

            base_name = Path(json_path).stem
            img_ext = Path(img_path).suffix

            dst_img = output_dir / "images" / split_name / f"{base_name}{img_ext}"
            dst_label = output_dir / "labels" / split_name / f"{base_name}.txt"

            shutil.copy2(img_path, dst_img)

            with open(dst_label, "w", encoding="utf-8") as f:
                for line in yolo_lines:
                    f.write(" ".join(f"{v:.6f}" for v in line) + "\n")

            stats[f"{split_name}_images"] += 1
            stats[f"{split_name}_labels"] += len(yolo_lines)

    # 生成 data.yaml
    yaml_path = output_dir / "data.yaml"
    abs_output = output_dir.resolve().as_posix()
    yaml_content = f"""path: {abs_output}
train: images/train
val: images/val

nc: {len(class_names)}
names: {class_names}
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("\n转换完成:")
    print(f"  输出目录: {output_dir}")
    print(f"  训练图片: {stats['train_images']}, 训练标注: {stats['train_labels']}")
    print(f"  验证图片: {stats['val_images']}, 验证标注: {stats['val_labels']}")
    print(f"  类别数: {len(class_names)}")
    print(f"  类别列表: {class_names}")
    print(f"  data.yaml: {yaml_path}")


if __name__ == "__main__":
    # ==================== 修改这里 ====================
    INPUT_DIR = r"D:\Project\HYJ_Pic\dab\train_data\foreign_matter\out_pred\2"   # 包含 .json 和图片的文件夹
    OUTPUT_DIR = r"D:\Project\HYJ_Pic\dab\train_data\foreign_matter\out_pred\2"   # 输出 YOLO 数据集目录

    VAL_RATIO = 0.2  # 验证集比例
    SEED = 42

    # 类别名列表。设为 None 则自动从 JSON 中收集
    CLASS_NAMES = None

    # 标签映射：可以把多个中文标签合并成一个类别
    # 例如：{"断胶": "ng", "气泡": "ng", "ok": "ok"}
    LABEL_MAPPING = None
    # =================================================

    build_dataset(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        val_ratio=VAL_RATIO,
        class_names=CLASS_NAMES,
        label_mapping=LABEL_MAPPING,
        seed=SEED,
    )
