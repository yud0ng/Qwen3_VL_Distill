# 在 kd_pipeline 目录下: make test | make smoke
.PHONY: test smoke smoke-deepspeed

test:
	python -m pytest tests/ -q

smoke:
	bash scripts/smoke_end_to_end.sh

smoke-deepspeed:
	bash scripts/smoke_deepspeed_zero2.sh
