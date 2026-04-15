# 在 kd_pipeline 目录下: make test | make smoke
.PHONY: test smoke

test:
	python -m pytest tests/ -q

smoke:
	bash scripts/smoke_end_to_end.sh
