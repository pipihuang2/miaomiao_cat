# miaomiao_cat

深度学习数据集处理的统一 CLI 工具集合。

本项目把原来零散的 60+ 个脚本重构为：

- `miaomiao`：可复用的 Python 包，提供 IO、图像、LabelMe JSON、CLI、日志等公共能力。
- `miaomiao.tools.*`：按领域组织的 CLI 工具入口。
- `mm-*`：高频工具的命令行快捷方式。

## 安装

```bash
# 基础能力（图像/JSON/YAML/数据集处理）
pip install -e ".[core,dev]"

# 需要深度学习相关依赖
pip install -e ".[ml,qr,dev]"
```

## 快速开始

```bash
# 统计 LabelMe JSON 标签数量
mm-count-labels -i /path/to/json_dir

# 批量 BMP 转 JPG
mm-bmp2jpg -i /path/to/bmp_dir -o /path/to/jpg_dir

# 计算文件 MD5
mm-md5 /path/to/file_or_dir
```

## 项目结构

```
.
├── pyproject.toml              # 包配置、依赖分组、CLI 入口
├── src/miaomiao/               # 统一基座包
│   ├── cli.py                  # argparse 辅助
│   ├── log.py                  # 统一日志
│   ├── io.py                   # 文件/JSON/YAML 操作
│   ├── image.py                # 图像公共操作
│   ├── labelme.py              # LabelMe JSON 解析
│   ├── dataset.py              # 数据集划分
│   └── tools/                  # CLI 工具入口
│       ├── image/
│       ├── labelme/
│       ├── dataset/
│       ├── calibration/
│       ├── model/
│       └── util/
├── tests/                      # 单元测试 + CLI smoke 测试
└── README.md
```

## 开发

```bash
# 运行测试
python -m pytest tests -v

# 代码检查与格式化
ruff check src tests
ruff format src tests
```

## 工具清单（已完成模板化示范）

| 命令 | 原脚本 | 说明 |
|------|--------|------|
| `mm-bmp2jpg` | `bmp_to_jpg.py` | 批量 BMP 转 JPG |
| `mm-md5` | `MD5/MD5_Check.py` | 计算文件/目录 MD5 |
| `mm-count-labels` | `coco/count_json_label_number.py` | 统计 LabelMe 标签数量 |

> 其余工具按 Phase 2 计划继续重构中。
