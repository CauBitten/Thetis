.PHONY: help preprocess train eval test clean

UV ?= uv
PYTHON ?= python

DATASET_DIR ?= dataset
DATA_DIR ?= data
SEED ?= 42

TRAIN_CONFIG ?= experiments/configs/stgcn_skeleton3d.yaml
CHECKPOINT ?= outputs/checkpoints/<run>/best.pt
TEST_PATH ?= tests/

help:
	@echo "Available targets:"
	@echo "  make preprocess  # build manifest, integrity report, and splits"
	@echo "  make train       # run spatiotemporal training"
	@echo "  make eval        # evaluate a checkpoint"
	@echo "  make test        # run test suite"
	@echo "  make clean       # remove generated artifacts"

preprocess:
	$(UV) run $(PYTHON) src/data/loader.py --input $(DATASET_DIR) --output $(DATA_DIR) --seed $(SEED)

train:
	$(UV) run $(PYTHON) src/models/spatiotemporal.py --config $(TRAIN_CONFIG)

eval:
	$(UV) run $(PYTHON) src/utils/metrics.py --checkpoint $(CHECKPOINT)

test:
	$(UV) run pytest $(TEST_PATH)

clean:
	$(UV) run $(PYTHON) -c "from pathlib import Path; import shutil; targets=['data/processed','data/splits','outputs/checkpoints','outputs/results','experiments/logs']; [shutil.rmtree(t, ignore_errors=True) for t in targets if Path(t).exists()]; print('clean: removed generated artifacts')"