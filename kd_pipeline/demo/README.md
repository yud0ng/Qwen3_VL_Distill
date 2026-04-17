# Demo · 2B 原始 vs 2B 蒸馏 vs 32B 参考

三列对比演示。32B 为预缓存，**演示无需 32B 硬件**。

## 目录
```
demo/
├── app.py                  # Gradio 三列 UI（XSS-safe via src/safe_html）
├── build_teacher_cache.py  # 从 teacher_responses.jsonl 精选缓存
├── teacher_cache.json      # 50 条 demo 问题（生成产物）
└── README.md               # 本文档
```

## 依赖

```bash
pip install "gradio>=4.0,<5.0"
# 加载 2B 模型还需：
pip install "transformers>=4.57" torch
```

## 生成缓存

基础用法（保留原图像路径）：
```bash
python demo/build_teacher_cache.py \
    --input ../teacher_responses.jsonl \
    --out demo/teacher_cache.json \
    --per_category 10
```

路径重写（推荐）：将集群路径 `/ocean/...` 改为本地路径：
```bash
python demo/build_teacher_cache.py \
    --input ../teacher_responses.jsonl \
    --out demo/teacher_cache.json \
    --per_category 10 \
    --image_root ./data/coco
```

默认产出 50 条（Count / Depth / Relational / Egocentric / Metric 各 10），
按类别 + L1/L2/L3 双标。

## 启动 UI

### UI 自检（不加载模型，无 GPU 也可）
```bash
python demo/app.py --no_load --port 7860
```

### C 交付 checkpoint 后完整演示
```bash
python demo/app.py \
    --original_model_path Qwen/Qwen3-VL-2B-Instruct \
    --distilled_model_path ../runs/variant_a_full_sft_general/adapter_final \
    --cache demo/teacher_cache.json \
    --port 7860
```

打开 http://127.0.0.1:7860。

## 功能

| 功能 | 说明 |
|---|---|
| 下拉预设问题 | `[category/level] question` 格式，自动填入问题 + 图像 + 32B 参考答案 |
| 自由输入 | 手动粘图 + 打问题；32B 列显示 `[cached reference only]` |
| 自动分层 | 每题显示 L1 / L2 / L3 标签 |
| 差异高亮 | 两个 2B 回答不同的 token 用 `<mark>` 包裹，XSS-safe |
| 健壮性 | 模型加载失败不阻断 UI；显示占位字符串，完整异常只在服务器日志中 |

## 安全说明

本 demo 设计为**本地演示**，默认绑定 `127.0.0.1`。不要在公网或共享服务器使用，原因：

1. `trust_remote_code=True` 用于 Qwen3-VL 加载 — 如 `--distilled_model_path`
   被恶意路径覆盖会执行任意代码
2. 虽然已做 HTML 转义 + CSV formula injection 防御，不代表抵御所有 Web 攻击

如需公网演示：
- 改用 `--host 0.0.0.0` 前请加反向代理 + Basic Auth
- `--distilled_model_path` 用硬编码或配置文件，不接受用户输入
- 考虑用容器隔离

## Gradio Demo 备案

演示硬件不可用时：
1. 提前录 3 分钟屏幕录屏
2. 导出 `teacher_cache.json` 的静态对比表（可用 pandas / HTML 模板）

## 即插即用契约

| C 交付什么 | 立即可演示 |
|---|---|
| `runs/variant_a_full_sft_general/adapter_final/` | `--distilled_model_path` 指向即可 |
| `runs/variant_bc_lora64/adapter_final/` | 同上 |
| 合并后全量权重（`scripts/export_merged_model.py`） | `--distilled_model_path exports/merged` |
