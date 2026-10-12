# YOLO 模型训练平台 · YOLO Forge

一个基于 Ultralytics 的可视化模型训练平台。通过浏览器创建和管理训练任务，查看训练进度、指标曲线、日志和模型权重，并对比不同实验的效果。

平台使用 FastAPI 调用服务器上的 `yolo` CLI 执行训练，使用 React Flow 展示训练画布，使用 ECharts 绘制指标曲线。当前以单服务器部署为主，支持本机多 GPU 训练。

## 主要功能

| 功能 | 说明 |
| --- | --- |
| 任务管理 | 新建、排队、启动、停止、断点续训和重新训练；拖动卡片调整显示顺序 |
| 克隆与重命名 | 克隆自动生成“原名称 - 副本 1、2…”；点击名称右侧的铅笔按钮修改任务名称 |
| 训练画布 | 展示数据集、基础模型、训练参数、指标、日志和训练产物，可双击对应节点查看或编辑 |
| 参数配置 | 表单编辑、粘贴参数导入、JSON 查看；显示识别结果及未识别参数提示 |
| 参数能力表 | 后端读取当前 Ultralytics 的 `default.yaml`，生成训练相关参数目录 |
| 数据集上传 | 选择本地 YOLO 检测数据集文件夹，校验目录、图片和标签，自动生成配置 |
| 实验对比 | 对比 2～4 次训练运行，展示参数差异、不同颜色的曲线、基准差值和指标优劣提示 |
| 训练历史 | 每次运行使用独立的 `Attempt` 编号，保留参数快照、指标、日志和权重 |
| 算力监控 | 查看当前服务器及 NVIDIA GPU 的资源信息 |
| 历史导入与回收站 | 导入已有 Ultralytics 训练结果，支持任务移入回收站、恢复和永久删除 |
| 界面主题 | 支持明亮、深色主题 |

调度器根据平台中活动任务占用的 GPU 安排训练，避免平台任务争抢同一设备；平台外启动的训练进程不属于该调度器的管理范围。

## 技术栈

- **前端**：React 19、TypeScript、Vite、React Flow、ECharts、Lucide Icons。
- **后端**：Python、FastAPI、SQLAlchemy、SQLite。
- **训练**：仓库中的 Ultralytics 源码、PyTorch、YOLO CLI。

## 项目结构

```text
.
├── README.md                 # 项目首页说明
├── ultralytics/              # 训练框架源码、配置及框架文档
├── web/
│   ├── frontend/             # Web 界面
│   ├── backend/              # API、数据库模型和训练调度器
│   ├── deploy/               # systemd 服务模板
│   ├── .env.example          # 环境变量配置示例
│   ├── README.md             # Web 平台补充说明
│   └── data/                 # 运行时生成的数据库、日志和训练输出
├── models/                   # 本地模型权重目录（按需准备）
├── pt/                       # 本地模型权重备用目录（按需准备）
└── 数据集/                   # 本地数据集及上传结果（按需准备）
```

模型权重、数据集、前端构建产物和平台运行数据需要在部署时准备或生成，不应依赖仓库提供本地训练数据。

## 安装与启动

### 1. 准备环境

建议使用 Linux、Python 3.11 及以上版本、Node.js 20 和 npm。使用 NVIDIA GPU 训练时，需要安装相应驱动及兼容的 PyTorch CUDA 版本，可通过 `nvidia-smi` 检查设备。

以下命令在已激活的 Python 或 Conda 环境中执行。请先根据训练服务器选择合适的 [PyTorch 安装方式](https://pytorch.org/get-started/locally/)，再安装项目依赖。

```bash
git clone https://github.com/scsg666g/yolo-xlpt.git
cd yolo-xlpt

# 安装仓库内的训练框架及平台后端
python -m pip install -e ./ultralytics
python -m pip install -e './web/backend[dev]'
```

### 2. 构建前端

在项目根目录执行：

```bash
cd web/frontend
npm ci
npm run build
cd ../..
```

构建结果位于 `web/frontend/dist/`，由后端直接提供访问。

### 3. 启动服务

在项目根目录执行：

```bash
# 使用当前 Python 环境安装的 YOLO CLI
export YOLO_BIN="$(command -v yolo)"

cd web/backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 18088 --workers 1
```

浏览器访问：

- 平台首页：`http://服务器地址:18088/`
- 算力资源：`http://服务器地址:18088/compute`
- API 文档：`http://服务器地址:18088/docs`

**保持 `--workers 1`**：训练调度器运行在 API 进程内，多进程启动会导致多个调度器同时运行。正式部署可参考 [systemd 服务模板](web/deploy/yolo-platform.service)，将其中的项目目录、Python 路径和配置文件路径改为实际值。

### 4. 配置服务器路径（可选）

默认项目路径从源码位置推导，默认数据保存到 `web/data/`。如果需要调整目录，可以参考 [环境变量示例](web/.env.example)。

| 环境变量 | 用途 |
| --- | --- |
| `YOLO_PROJECT_ROOT` | 项目根目录 |
| `ULTRALYTICS_ROOT` | 仓库内 Ultralytics 目录 |
| `YOLO_BIN` | 训练环境中 `yolo` 可执行文件的绝对路径 |
| `PLATFORM_DATA_ROOT` | 平台运行数据根目录 |
| `PLATFORM_RUNS_ROOT` | 训练结果保存目录 |
| `PLATFORM_RUNTIME_ROOT` | 训练运行时配置目录 |
| `PLATFORM_LOG_ROOT` | 平台日志目录 |
| `DATABASE_URL` | 数据库连接地址，默认使用 SQLite |
| `NVIDIA_SMI_BIN` | `nvidia-smi` 命令或绝对路径 |
| `PLATFORM_MKL_THREADING_LAYER` | 训练进程 MKL 线程设置，默认 `GNU` |

示例文件包含原开发服务器的路径，部署到其他位置时应先调整。直接启动 Uvicorn 不会自动读取 `web/.env`；需要将变量导入环境，或由 systemd 的 `EnvironmentFile` 加载。

## 数据集准备

### 通过网页上传

在“新建训练任务”窗口的数据集区域点击“上传数据集”，选择包含以下结构的最外层文件夹：

```text
火焰数据集/
├── images/
│   ├── train/                # 训练图片
│   └── val/                  # 验证图片
└── labels/
    ├── train/                # 对应训练标签 .txt
    └── val/                  # 对应验证标签 .txt
```

当前上传入口支持 **YOLO 目标检测数据集**。图片与标签使用相同的文件名，例如 `images/train/001.jpg` 对应 `labels/train/001.txt`。

标签每行表示一个目标：

```text
类别编号 中心x 中心y 宽 高
0 0.50 0.50 0.20 0.30
```

类别编号从 `0` 开始，坐标与尺寸归一化到 `0～1`，宽高必须大于 `0`。类别名称可以在上传窗口填写，也可从目录中的 `data.yaml`、`data.yml` 或 `classes.txt` 读取。缺少标签或空标签的图片可作为背景图，完成检查时会显示提醒。

上传后保留原文件夹名称，保存到 `数据集/<原文件夹名>/`，自动生成 `data.yaml` 并选中该数据集。配置中的 `path` 使用服务器上的实际绝对路径。同名目录会拒绝上传，避免覆盖已有数据。

支持 JPG、JPEG、PNG、BMP、WEBP、TIF、TIFF；单张图片最大 256 MB，标签或配置文件最大 4 MB，单次上传最大 50 GB、20 万个文件。详细的格式要求、暂停和重试说明见 [Web 平台文档](web/README.md#上传数据集)。

### 使用服务器已有数据集

将数据集配置放在项目的 `数据集/` 或 `ultralytics/xl/` 下，在创建任务时选择对应 YAML 文件。例如：

```yaml
path: /服务器上的数据集目录/火焰数据集
train: images/train
val: images/val
names:
  0: fire
```

将 `path` 替换为实际路径。基础模型权重可放在 `models/`、`pt/` 或 `ultralytics/weights/` 下，供平台发现并选择。

## 使用流程

1. 打开“算力资源”，确认服务器和 GPU 状态。
2. 新建训练任务，填写名称，选择模型、数据集、GPU 和训练参数。
3. 启动任务，进入训练画布查看进度、曲线、日志及训练产物。
4. 需要调整方案时，克隆任务并修改配置；名称旁的铅笔按钮可随时重命名。
5. 在“实验对比”中选择不同运行，查看指标和参数差异。判断训练效果时，应同时考虑数据集、验证方式及训练是否完成。

任务名称是展示名称，重命名不会移动权重、修改训练参数或改变已有日志路径。

## 训练结果与历史

平台创建的每次训练运行使用独立目录：

```text
web/data/runs/<任务类型>/<任务ID>/attempt-001/
web/data/runs/<任务类型>/<任务ID>/attempt-002/
```

每个 `Attempt` 保留自身的训练参数快照、指标和产物。平台也可以导入 `ultralytics/runs/` 下已有的训练记录；历史导入不会移动原训练文件。

默认数据库为 `web/data/platform.db`。迁移部署时，应一起备份数据库、训练结果、日志，以及模型和数据集文件，并核对配置中的绝对路径。

## 开发与维护

前端开发可在 `web/frontend/` 执行 `npm run dev`；开发服务器默认将 `/api` 请求代理到 `http://127.0.0.1:18088`，因此仍需启动后端。

后端测试在 `web/backend/` 执行：

```bash
python -m pytest
```

修改前端后重新执行 `npm run build`；修改后端后重启服务。平台目前管理本机训练任务，尚未提供添加远程服务器或跨服务器统一调度的功能。当前未内置登录认证，如需开放给其他用户，可在部署网关配置访问认证。

## 相关文档

- [Web 平台说明](web/README.md)
- [Ultralytics 框架说明](ultralytics/README.md)
- [Ultralytics 许可证](ultralytics/LICENSE)

仓库内 Ultralytics 组件的许可证信息以其说明及许可证文件为准。
