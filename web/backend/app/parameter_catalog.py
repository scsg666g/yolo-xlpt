from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .config import ULTRALYTICS_ROOT


DEFAULT_CFG_PATH = ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "default.yaml"

# These values are controlled by the platform and must never be imported as a
# free-form training override.
PLATFORM_KEYS = {
    "task", "mode", "model", "data", "project", "name", "device", "epochs",
    "batch", "imgsz", "workers", "optimizer", "exist_ok", "resume", "save_dir",
    "source",
}

# The YAML file groups training-relevant settings into these sections. Keeping
# the section detection here means newly added keys in the installed Ultralytics
# version are picked up automatically without another frontend/backend edit.
SECTION_RE = re.compile(r"^#\s+(Train settings|Segmentation|Classification|Val/Test settings|Hyperparameters)\b")
MAJOR_SECTION_RE = re.compile(r"^#\s+[^-].*-{5,}\s*$")
KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):(?:\s|$)")

LABELS = {
    "epochs": "训练轮数", "time": "最长训练时长", "patience": "早停耐心值", "batch": "Batch 大小",
    "imgsz": "输入尺寸", "save": "保存训练检查点", "save_period": "权重保存周期", "cache": "缓存数据",
    "workers": "数据线程", "pretrained": "预训练权重", "cls_remap": "按类别名重映射分类头",
    "optimizer": "优化器", "verbose": "输出详细训练日志", "seed": "随机种子", "deterministic": "确定性训练",
    "single_cls": "所有类别合并为单类", "rect": "矩形训练", "cos_lr": "余弦学习率",
    "classes": "仅训练指定类别 ID",
    "close_mosaic": "关闭 Mosaic 轮数", "amp": "AMP", "fraction": "数据集比例", "profile": "分析模型性能",
    "freeze": "冻结层", "multi_scale": "多尺度训练", "compile": "启用 torch.compile",
    "channels_last": "Channels-last 内存格式", "overlap_mask": "合并实例掩码", "mask_ratio": "掩码下采样比率",
    "dropout": "Dropout 比率", "val": "训练期间运行验证", "split": "验证数据集划分",
    "save_json": "保存 COCO JSON 结果", "conf": "验证置信度阈值", "iou": "验证 NMS IoU 阈值",
    "max_det": "单张图最大检测数", "quantize": "量化精度", "dnn": "使用 OpenCV DNN", "plots": "保存训练与验证图表",
    "nms": "NMS 设置", "lr0": "初始学习率", "lrf": "最终学习率系数", "momentum": "动量",
    "weight_decay": "权重衰减", "warmup_epochs": "预热轮数", "warmup_momentum": "预热初始动量",
    "warmup_bias_lr": "预热偏置学习率", "distill_model": "知识蒸馏教师模型", "dis": "蒸馏损失权重",
    "box": "Box loss 权重", "cls": "Cls loss 权重", "cls_pw": "类别不平衡权重指数", "dfl": "DFL loss 权重",
    "pose": "Pose loss 权重", "kobj": "关键点置信度权重", "rle": "RLE loss 权重", "angle": "角度 loss 权重",
    "dlog": "深度 SILog loss 权重", "dgrad": "深度梯度 loss 权重", "dlam": "深度尺度不变系数",
    "nbs": "损失归一化标称 Batch", "hsv_h": "HSV 色相增强", "hsv_s": "HSV 饱和度增强", "hsv_v": "HSV 明度增强",
    "degrees": "旋转角度", "translate": "平移增强", "scale": "缩放增强", "shear": "剪切增强",
    "perspective": "透视增强", "flipud": "上下翻转概率", "fliplr": "左右翻转概率", "bgr": "RGB/BGR 通道交换概率",
    "mosaic": "Mosaic 概率", "mixup": "MixUp 概率", "cutmix": "CutMix 增强概率", "copy_paste": "Copy-paste 概率",
    "copy_paste_mode": "Copy-paste 策略", "auto_augment": "自动增强策略", "erasing": "随机擦除概率",
}

# Null defaults in default.yaml do not carry enough type information. These
# overrides preserve the types accepted by the Ultralytics CLI.
TYPE_OVERRIDES = {
    "time": "number", "conf": "number", "distill_model": "text", "quantize": "text",
    "nms": "boolean", "channels_last": "boolean", "dnn": "boolean",
}

# Some options live outside the train/val blocks in default.yaml but are also
# consumed by model.train(). Keep these explicit exceptions small and sourced
# from the same YAML file.
CROSS_SECTION_TRAINING_KEYS = {"classes"}


def _value_type(value: Any, key: str) -> str:
    if key in TYPE_OVERRIDES:
        return TYPE_OVERRIDES[key]
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "text"


def _section_keys(text: str) -> list[tuple[str, str]]:
    section = ""
    keys: list[tuple[str, str]] = []
    for line in text.splitlines():
        heading = SECTION_RE.match(line.strip())
        if heading:
            section = heading.group(1)
            continue
        if MAJOR_SECTION_RE.match(line.strip()):
            section = ""
            continue
        match = KEY_RE.match(line)
        if section and match:
            key = match.group(1).lower()
            if key not in PLATFORM_KEYS:
                keys.append((key, section))
    return keys


def load_parameter_catalog() -> dict[str, dict[str, Any]]:
    """Read the installed Ultralytics training parameter surface."""
    if not DEFAULT_CFG_PATH.exists():
        return {}
    text = DEFAULT_CFG_PATH.read_text(encoding="utf-8")
    values = yaml.safe_load(text) or {}
    catalog: dict[str, dict[str, Any]] = {}
    for key, section in _section_keys(text):
        if key in catalog:
            continue
        value = values.get(key)
        catalog[key] = {
            "key": key,
            "label": LABELS.get(key, key),
            "type": _value_type(value, key),
            "default": value,
            "section": section,
        }
    for key in CROSS_SECTION_TRAINING_KEYS:
        if key in values and key not in catalog:
            value = values[key]
            catalog[key] = {
                "key": key,
                "label": LABELS.get(key, key),
                "type": _value_type(value, key),
                "default": value,
                "section": "Training-compatible",
            }
    return catalog


def training_parameter_keys() -> set[str]:
    return set(load_parameter_catalog())
