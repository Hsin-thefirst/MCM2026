PYTHON ?= .venv/bin/python
RUN ?= benchmark/outputs/development_consolidated
PACKAGE ?= dist/SafeInvest_submission.zip

.PHONY: help check test package verify-package restore clean guide benchmark report audit replay
help:
	@echo 'make check      检查冻结程序、环境及交付资料；不运行最终评价'
	@echo 'make test       运行统一测试，不读取真实最终留出集'
	@echo 'make package    打包当前源码、讲解与数值证据，默认输出 $(PACKAGE)'
	@echo 'make verify-package  核验 PACKAGE 指定ZIP的文件hash与外部收据'
	@echo 'make restore PACKAGE=<可信ZIP>  仅恢复保存数值证据，不覆盖不同文件'
	@echo 'make clean      仅清理Python/绘图缓存，保留环境、结果和交付包'
	@echo 'make guide      从保存数值重建中文讲解，缺大证据时先restore'
	@echo 'make benchmark  运行开发集实验；新实验请指定 RUN'
	@echo 'make report     从保存结果重新生成中文报告'
	@echo 'make audit      核验账本与交叉验证划分'
	@echo 'make replay     独立重放开发集收益'

test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:benchmark $(PYTHON) -m unittest discover -s tests -v

check:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/project_tools.py check

package:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/project_tools.py package --archive "$(PACKAGE)"

verify-package:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/project_tools.py verify --archive "$(PACKAGE)"

restore:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/project_tools.py restore --archive "$(PACKAGE)"

clean:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/project_tools.py clean

guide:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/build_solution_guide.py --research-run benchmark/outputs/development_customer_risk_review_v1 --out docs/SOLUTION_GUIDE.md

benchmark:
	$(PYTHON) benchmark/run.py development --out $(RUN)

report:
	$(PYTHON) benchmark/run.py report --out $(RUN)

audit:
	$(PYTHON) benchmark/run.py audit --out $(RUN)

replay:
	$(PYTHON) scripts/verify_benchmark.py $(RUN)
