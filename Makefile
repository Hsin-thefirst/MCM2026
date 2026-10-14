PYTHON ?= .venv/bin/python
RUN ?= benchmark/outputs/development_consolidated
CONFIG ?= configs/benchmark.json

.PHONY: help test benchmark study report audit replay
help:
	@echo 'make test       运行统一测试，不读取真实最终留出集'
	@echo 'make benchmark  运行开发集实验；新实验请指定 RUN'
	@echo 'make study      运行配对的路径风险选择研究；新实验请指定 RUN'
	@echo 'make report     从保存结果重新生成中文报告'
	@echo 'make audit      核验账本与交叉验证划分'
	@echo 'make replay     独立重放开发集收益'

test:
	$(PYTHON) scripts/run_tests.py

benchmark:
	$(PYTHON) benchmark/run.py development --config $(CONFIG) --out $(RUN)

study: CONFIG = configs/path_risk.json
study: RUN = benchmark/outputs/development_path_risk_study
study:
	$(PYTHON) scripts/run_path_risk_study.py --config $(CONFIG) --out $(RUN)

report:
	$(PYTHON) benchmark/run.py report --out $(RUN)

audit:
	$(PYTHON) benchmark/run.py audit --out $(RUN)

replay:
	$(PYTHON) scripts/verify_benchmark.py $(RUN)
