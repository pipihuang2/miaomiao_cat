import os
import json

def process_json_files(directory, image_extensions):
    """
    遍历目录中的 JSON 文件，检查和更新其中的 imagePath。
    """
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')] # endswith 是一个判断后缀的
    for json_file in json_files:
        json_path = os.path.join(directory, json_file)

        # 加载 JSON 文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image_path = data.get("imagePath", "")
        image_name_without_ext = os.path.splitext(image_path)[0]

        # 查找匹配的图片文件
        matching_images = [
            f for f in os.listdir(directory)
            if os.path.splitext(f)[0] == image_name_without_ext and os.path.splitext(f)[1].lower() in image_extensions
        ]

        if matching_images:
            # 更新 JSON 中的 imagePath 为匹配的图片
            updated_image_path = matching_images[0]
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
    target_directory = r"F:\deeplearning\pytorch\miao_tools\miaomiao_cat\json"  # 替换为实际目录路径
    image_extensions = ['.jpg', '.jpeg', '.bmp', '.png']  # 支持的图片格式

    # 执行处理
    process_json_files(target_directory, image_extensions)
