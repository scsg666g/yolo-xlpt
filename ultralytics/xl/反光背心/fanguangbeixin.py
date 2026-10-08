import os  # 导入操作系统模块，用于设置训练进程环境变量

os.environ["MKL_THREADING_LAYER"] = "GNU"  # 使用 GNU 线程层，避免 MKL 与 OpenMP 等并行库发生冲突

from ultralytics import YOLO  # 导入 Ultralytics YOLO 训练接口


model = YOLO("/data/yolo/yolo/models/yolo26x.pt")  # 加载 YOLO26x 预训练权重，在此基础上微调反光背心模型

model.train(  # 开始目标检测训练
    data="/data/yolo/yolo/ultralytics/xl/反光背心/fanguangbeixin.yaml",  # 数据集 YAML：定义训练集、验证集和类别名称
    epochs=80,  # 最大训练 80 轮；若验证指标长期无提升，可能由 patience 提前停止
    imgsz=960,  # 训练输入尺寸；较大尺寸有利于小目标，但会明显增加显存占用和训练时间
    batch=10,  # 全局批量大小；双 GPU DDP 时通常每张卡约处理 5 张图片
    device="0,1",  # 使用编号为 0 和 1 的两张 GPU 进行分布式训练
    workers=8,  # 每个 DDP 训练进程的数据加载进程数；双卡实际可能产生约 16 个 worker
    cache="disk",  # 把预处理后的图片缓存到磁盘，后续 epoch 读取更快但会占用磁盘空间
    amp=True,  # 开启自动混合精度，通常可降低显存占用并提高训练速度
    optimizer="MuSGD",  # 使用 YOLO26 推荐的 MuSGD 优化器
    lr0=0.00038,  # 初始学习率；适合大型预训练模型的小步微调
    lrf=0.882,  # 最终学习率系数；训练末期学习率约为 lr0 × lrf
    momentum=0.948,  # 优化器动量，平滑梯度更新并帮助收敛
    weight_decay=0.00027,  # 权重衰减，抑制模型过拟合
    warmup_epochs=1.0,  # 前 1 轮逐步提升学习率，减少训练初期震荡
    patience=25,  # 验证指标连续 25 轮没有提升时触发早停
    hsv_h=0.013,  # 色调增强幅度；保持较小以减少反光背心颜色失真
    hsv_s=0.35,  # 饱和度增强幅度，适应不同相机、天气和色彩条件
    hsv_v=0.20,  # 亮度增强幅度，适应阴影、夜间和曝光变化
    degrees=2.0,  # 随机旋转范围为 ±2°，模拟轻微摄像头倾斜
    translate=0.15,  # 随机平移比例，提升目标位于画面不同位置时的鲁棒性
    scale=0.60,  # 随机缩放幅度，增强近距离与远距离目标的尺度适应能力
    shear=0.0,  # 关闭剪切变换，避免产生不自然的背心形状
    perspective=0.0,  # 关闭透视变换，避免引入与真实监控场景不符的畸变
    fliplr=0.5,  # 以 50% 概率水平翻转图片，增强左右方向泛化能力
    flipud=0.0,  # 禁止上下翻转，避免生成倒置人员等不真实样本
    mosaic=0.90,  # 以 90% 概率使用 Mosaic，增加背景组合和目标尺度多样性
    mixup=0.05,  # 以 5% 概率使用 MixUp，提供轻量正则化并降低过拟合
    close_mosaic=15,  # 最后 15 轮关闭 Mosaic，让模型重新适应真实单图分布
    seed=42,  # 固定随机种子，便于复现实验结果
    project="/data/yolo/yolo/ultralytics/runs/detect",  # 训练结果的根目录
    name="反光背心_yolo26x_960",  # 本次实验名称，同时作为结果子目录名称
    exist_ok=False,  # 不覆盖已有同名实验目录；重跑时会自动创建带编号的新目录
    plots=True,  # 输出训练曲线、PR/F1 曲线、混淆矩阵和验证样例图
)
