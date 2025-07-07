import os
import json
import tqdm


def process_json_files(directory, image_extensions):
    """
    遍历目录中的 JSON 文件，检查和更新其中的 imagePath。
    """
    # 一次性获取目录中所有文件
    all_files = os.listdir(directory)

    # 预先构建图片文件的映射：基础文件名 -> 图片文件名
    image_mapping = {}
    for f in all_files:
        ext = os.path.splitext(f)[1].lower()
        if ext in image_extensions:
            base = os.path.splitext(f)[0]
            # 如果存在多个同名图片，只保留第一个匹配
            if base not in image_mapping:
                image_mapping[base] = f

    # 筛选出所有 JSON 文件
    json_files = [f for f in all_files if f.endswith('.json')]

    for json_file in tqdm.tqdm(json_files):
        json_path = os.path.join(directory, json_file)

        # 加载 JSON 文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image_path = data.get("imagePath", "")
        image_name_without_ext = os.path.splitext(image_path)[0]

        # 使用预构建的映射查找对应的图片文件
        if image_name_without_ext in image_mapping:
            updated_image_path = image_mapping[image_name_without_ext]
            if image_path != updated_image_path:
                print(f"更新 {json_file} 的 imagePath: {image_path} -> {updated_image_path}")
                data["imagePath"] = updated_image_path

                # 保存更新后的 JSON 文件
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
        else:
            print(f"未找到与 {json_file} 中 imagePath 匹配的图片文件：{image_path}")


# 主程序入口
if __name__ == "__main__":
    # 设置目标目录路径和支持的图片格式
    target_directory = r"G:\data\F2_New\val"  # 替换为实际目录路径
    image_extensions = ['.jpg', '.jpeg', '.bmp', '.png']  # 支持的图片格式

    # 执行处理
    process_json_files(target_directory, image_extensions)
