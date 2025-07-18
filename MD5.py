import hashlib

def calculate_md5(file_path, chunk_size=8192):
    md5 = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()
    except FileNotFoundError:
        print(f"文件未找到：{file_path}")
        return None

# 示例用法
file_path = r"\\tianwei\TianweiNas\NAS\cosmos\models\DAB_models\yolo8s_for_seam\20250701\seamdetector.onnx"  # 替换为你要计算的文件路径
md5_value = calculate_md5(file_path)
if md5_value:
    print(f"{file_path} 的 MD5 值是：{md5_value}")