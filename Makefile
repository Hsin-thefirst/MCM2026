PYTHON ?= .venv/bin/python
RUN ?= benchmark/outputs/development_consolidated

.PHONY: help test benchmark report audit replay
help:
	@echo 'make test       运行统一测试，不读取真实最终留出集'
	@echo 'make benchmark  运行开发集实验；新实验请指定 RUN'
	@echo 'make report     从保存结果重新生成中文报告'
	@echo 'make audit      核验账本与交叉验证划分'
	@echo 'make replay     独立重放开发集收益'

test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:benchmark $(PYTHON) -m unittest discover -s tests -v

benchmark:
	$(PYTHON) benchmark/run.py development --out $(RUN)

report:
	$(PYTHON) benchmark/run.py report --out $(RUN)

audit:
	$(PYTHON) benchmark/run.py audit --out $(RUN)

replay:
	$(PYTHON) scripts/verify_benchmark.py $(RUN)
