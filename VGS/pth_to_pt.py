import torch

# 读取原始 .torch 文件
state_dict = torch.load(r"D:\Project\HYJ\sam2_train\fine-tune-train_segment_anything_2_in_60_lines_of_code\checkpoints\model.torch", map_location="cpu")

# 重新保存为 .pt 格式
torch.save({"model": state_dict}, "model.pt")

print("✅ 已将 model.torch 转存为 model.pt（包含 'model' 键）")