# YOLO 模型训练平台

独立于 Ultralytics 源码的训练任务控制台。平台通过受控参数调用现有 `yolo` CLI，并读取标准的 `args.yaml`、`results.csv` 和权重文件，因此无需修改训练框架。

## 功能

- 训练任务创建、排队、启动、停止、克隆和历史任务导入
- 每条训练任务对应一张 React Flow 无限画布
- 实时训练进度、指标曲线、日志、GPU 设备和训练产物
- GPU 独占调度，防止多个任务占用同一设备
- 训练参数白名单和模型/数据集目录边界检查
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

