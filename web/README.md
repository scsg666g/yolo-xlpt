# YOLO 模型训练平台

独立于 Ultralytics 源码的训练任务控制台。平台通过受控参数调用现有 `yolo` CLI，并读取标准的 `args.yaml`、`results.csv` 和权重文件，因此无需修改训练框架。

## 功能

- 训练任务创建、排队、启动、停止、克隆和历史任务导入
- 每条训练任务对应一张 React Flow 无限画布
- 实时训练进度、指标曲线、日志、GPU 设备和训练产物
- GPU 独占调度，防止多个任务占用同一设备
- 训练参数白名单和模型/数据集目录边界检查
- 本地数据集文件夹上传、格式检查、自动生成配置和选中数据集
- 服务重启后的任务中断识别

## 目录

- `frontend/`：React、TypeScript、React Flow 和 ECharts
- `backend/`：FastAPI、SQLAlchemy 和训练调度器
- `data/`：SQLite 数据库、日志和平台训练产物，不提交 Git
- `deploy/`：systemd 服务模板

## 安装与运行

```bash
cd /data/yolo-模型训练平台开发/yolo/web
cp .env.example .env

/data/conda/envs/yolo/bin/python -m pip install -e './backend[dev]'

cd frontend
npm install
npm run build

cd ../backend
/data/conda/envs/yolo/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 18088 --workers 1
```

打开 `http://服务器地址:18088`。API 文档位于 `/docs`。

> 当前调度器运行在 API 进程内，因此 Uvicorn 必须保持 `--workers 1`。若以后部署多个 API 实例，应将调度器拆分为独立 Worker，并使用 PostgreSQL/Redis 做协调。

## 上传数据集

在「创建训练任务 → 数据集配置」旁点击「上传数据集」，选择最外层文件夹。目前支持 YOLO 目标检测，窗口中会显示目录格式、标签示例、类别编号和文件限制：

```text
我的数据集/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

图片与标签应同名，标签每行格式为 `类别编号 中心x 中心y 宽 高`，坐标归一化到 0～1，宽高大于 0。类别名称每行填写一个，对应编号从 0 开始；也可从上传目录中的 `data.yaml` / `data.yml` 的 `names`，或 UTF-8 编码的 `classes.txt` 读取。无标签或空标签图片可作为背景图，完成后会显示提醒。

服务器保留原文件夹名称（支持中文），保存到项目的 `数据集/<原文件夹名>/`。系统自动生成 `data.yaml`，其中 `path` 是该文件夹的实际绝对路径，`train` / `val` 分别为 `images/train` / `images/val`；原配置中的其他服务器路径会被替换。上传完成后自动选中该数据集，同名目录会拒绝上传，不覆盖已有内容。

支持 JPG、JPEG、PNG、BMP、WEBP、TIF、TIFF，单张图片最大 256 MB，单个标签或配置文件最大 4 MB，总大小最大 50 GB / 20 万个文件。浏览器不能传递空目录，因此缺失的空标签目录会自动补齐。系统文件和缓存会忽略；其他不符合格式的文件会提示整理。建议用 Chrome 或 Edge 选择文件夹。

上传窗口保持打开时，可以暂停或在连接中断后重试，已经传完的文件无需重复上传。关闭窗口会清理临时上传，刷新页面后需要重新选择；超过 24 小时未使用的上传记录会在后续上传时清理。数据在图片、标签和类别检查通过后才会加入可选数据集，完成响应丢失时重试不会重复保存。

## 训练目录

新任务默认写入：

```text
web/data/runs/<task-type>/<task-id>/
```

已有的 `ultralytics/runs/**/args.yaml` 会自动以只读方式导入。导入不会移动、删除或改写历史训练文件。

## 安全边界

- 模型只能从项目的 `models/`、`pt/` 或 `ultralytics/weights/` 选择。
- 数据集配置只能从项目的 `数据集/` 或 `ultralytics/xl/` 选择。
- 训练命令使用参数数组执行，不调用 Shell。
- 产物下载必须位于对应任务的训练目录内。
- 面向非受信网络部署前，应在 Nginx 或统一网关增加身份认证和 HTTPS。
