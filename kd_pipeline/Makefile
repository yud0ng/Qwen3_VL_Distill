# 在 kd_pipeline 目录下:
#   make test         — 全部 pytest
#   make smoke        — 训练冒烟（dummy 数据）
#   make d_pipeline   — D 角色全链路：logit subset + teacher cache + e2e tests
#   make d_cache      — 仅重建 demo/teacher_cache.json
#   make d_logit_ids  — 仅产出 data/logit_subset_ids.txt
#   make coverage     — 带覆盖率报告
#   make lint         — bandit 安全扫描（若已安装）
.PHONY: test smoke smoke-deepspeed d_pipeline d_cache d_logit_ids coverage lint help

TEACHER_JSONL ?= ../teacher_responses.jsonl
IMAGE_ROOT ?= ./data/coco

help:
	@echo "available targets: test smoke smoke-deepspeed d_pipeline d_cache d_logit_ids coverage lint"

test:
	python -m pytest tests/ -q

smoke:
	bash scripts/smoke_end_to_end.sh

smoke-deepspeed:
	bash scripts/smoke_deepspeed_zero2.sh

d_logit_ids:
	python scripts/select_logit_subset.py \
		--input $(TEACHER_JSONL) \
		--n 10000 \
		--out_ids data/logit_subset_ids.txt \
		--out_manifest data/logit_subset_ids.manifest.json

d_cache:
	python demo/build_teacher_cache.py \
		--input $(TEACHER_JSONL) \
		--out demo/teacher_cache.json \
		--per_category 10 \
		--image_root $(IMAGE_ROOT)

d_pipeline: d_logit_ids d_cache
	python -m pytest tests/test_e2e_d_pipeline.py -v
	@echo ""
	@echo "D pipeline ready. Deliverables:"
	@echo "  - data/logit_subset_ids.txt       (for A's --id_list)"
	@echo "  - demo/teacher_cache.json         (for demo/app.py)"
	@echo ""

coverage:
	python -m pytest tests/ \
		--cov=src --cov=scripts --cov=demo \
		--cov-report=term-missing --cov-report=html

lint:
	bandit -r src/ scripts/ demo/ -ll -q || true
