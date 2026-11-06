# YOLO 目标检测可视化系统

> 课程：深度学习综合实践  
> 作者：安小刚  
> 环境：Windows 11 + PyCharm + Miniconda（环境名 `yolo_39`）

---

## 一、功能介绍

本程序基于 **Ultralytics YOLO（2026 最新版 ultralytics 8.4.x）** 构建了一个图形化目标检测界面，主要功能：

| 功能 | 说明 |
| --- | --- |
| 导入图片 | 单张图片检测，支持 jpg/png/bmp |
| 导入视频 | 逐帧实时检测，支持 mp4/avi/mov/mkv |
| 打开摄像头 | 调用本机摄像头实时检测 |
| 置信度阈值调节 | 滑动条 0.01~0.99，实时生效 |
| IOU 阈值调节 | 滑动条 0.01~0.99 |
| 模型权重切换 | 自动扫描 `weights/` 目录下的 `.pt` 文件 |
| 结果统计 | 右侧面板实时显示每类目标数量 |
| 手动保存结果 | 一键把当前检测画面保存到 `output/` 目录 |
| GPU 加速 | 自动使用 CUDA（torch 2.7.0+cu118） |

---

## 二、目录结构

```
yolo_vision_app/
├── main.py              # 主程序（GUI + 检测逻辑）
├── 启动程序.bat          # Windows 一键启动脚本（双击即可）
├── README.md            # 使用说明
├── weights/             # 存放 YOLO 权重文件（.pt）
│   └── yolo11n.pt       # 首次运行自动下载
└── output/              # 检测结果图片自动保存到此目录
```

---

## 三、运行方法

### 方法 1：双击 `启动程序.bat`（推荐）

脚本会自动激活 `yolo_39` 环境并运行 `main.py`。

### 方法 2：在 PyCharm 中运行

1. 打开 PyCharm → 打开文件夹 `E:\Desktop\深度学习综合实践\yolo_vision_app`
2. 右下角选择解释器 → 添加解释器 →  Conda 环境 → 现有环境 → 选择 `yolo_39`
3. 打开 `main.py` → 右键 Run

### 方法 3：命令行

```bat
conda activate yolo_39
cd /d E:\Desktop\深度学习综合实践\yolo_vision_app
python main.py
```

---

## 四、首次运行说明

首次启动时，程序会自动从 Ultralytics 官方仓库下载 `yolo11n.pt`（约 5 MB）到 `weights/` 目录。  
如果下载失败，请手动从 https://github.com/ultralytics/assets/releases 下载后放到 `weights/` 目录。

---

## 五、如何使用自己训练的权重

如果你已经按作业要求训练了自己的数据集（4 类以上目标），把训练得到的 `best.pt` 复制到 `weights/` 目录，然后在右侧面板：

1. 点击「刷新权重列表」
2. 选中 `best.pt`
3. 点击「加载选中权重」

之后导入图片/视频检测时就会使用你自己训练的模型。

---

## 六、操作步骤（对应作业要求）

1. **配置环境**：已完成（conda 环境 `yolo_39`）
2. **准备数据**：自行录制 ≥30 秒、包含 ≥4 类目标的视频
3. **标注数据**：使用 LabelImg / Roboflow 标注
4. **训练**：`yolo detect train data=你的数据集.yaml model=yolo11n.pt epochs=100`
5. **测试**：把 `best.pt` 放到 `weights/`，运行本程序导入视频测试
6. **保存结果**：点击「保存当前结果」按钮，截图放到实验报告中
