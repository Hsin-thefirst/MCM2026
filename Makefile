ifeq ($(OS),Windows_NT)
PYTHON ?= .venv/Scripts/python.exe
else
PYTHON ?= .venv/bin/python
endif
RUN ?= benchmark/outputs/development_consolidated
CONFIG ?= configs/benchmark.json

.PHONY: help test benchmark study efficiency learning neural report audit replay
help:
	@echo 'make test       运行统一测试，不读取真实最终留出集'
	@echo 'make benchmark  运行开发集实验；新实验请指定 RUN'
	@echo 'make study      运行配对的路径风险选择研究；新实验请指定 RUN'
	@echo 'make efficiency 运行固定目标与情景效用效率研究；新实验请指定 RUN'
	@echo 'make learning   运行反事实离线强化学习研究；新实验请指定 RUN'
	@echo 'make neural     运行小型神经网络路径约束研究；新实验请指定 RUN'
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

efficiency: CONFIG = configs/efficiency.json
efficiency: RUN = benchmark/outputs/development_efficiency
efficiency:
	$(PYTHON) benchmark/run.py development --config $(CONFIG) --out $(RUN)

learning: CONFIG = configs/learning.json
learning: RUN = benchmark/outputs/development_counterfactual_fqi_v2
learning:
	$(PYTHON) benchmark/run.py development --config $(CONFIG) --out $(RUN)

neural: CONFIG = configs/neural.json
neural: RUN = benchmark/outputs/development_neural_path
neural:
	$(PYTHON) benchmark/run.py development --config $(CONFIG) --out $(RUN)

report:
	$(PYTHON) benchmark/run.py report --out $(RUN)

audit:
	$(PYTHON) benchmark/run.py audit --out $(RUN)

replay:
	$(PYTHON) scripts/verify_benchmark.py $(RUN)
