.PHONY: help preprocess episodes train eval test clean

UV ?= uv
PYTHON ?= python

DATASET_DIR ?= dataset
DATA_DIR ?= data
SEED ?= 42

MANIFEST ?= data/processed/manifest.csv
EPISODES_DIR ?= data/episodes/
N_WAY ?= 5
K_SHOT ?= 5
Q_QUERY ?= 15
EPISODES_PER_SPLIT ?= 1000

TRAIN_CONFIG ?= experiments/configs/stgcn_skeleton3d.yaml
CHECKPOINT ?= outputs/checkpoints/<run>/best.pt
TEST_PATH ?= tests/

help:
	@echo "Available targets:"
	@echo "  make preprocess  # build manifest, integrity report, and splits"
	@echo "  make episodes    # sample N-way K-shot episodes from the manifest"
	@echo "  make train       # run spatiotemporal training"
	@echo "  make eval        # evaluate a checkpoint"
	@echo "  make test        # run test suite"
	@echo "  make clean       # remove generated artifacts"

preprocess:
	$(UV) run $(PYTHON) src/data/loader.py --input $(DATASET_DIR) --output $(DATA_DIR) --seed $(SEED)

episodes:
	$(UV) run $(PYTHON) src/data/episode_sampler.py \
		--manifest $(MANIFEST) \
		--output $(EPISODES_DIR) \
		--n-way $(N_WAY) --k-shot $(K_SHOT) --q-query $(Q_QUERY) \
		--episodes-per-split $(EPISODES_PER_SPLIT) \
		--seed $(SEED)

train:
	$(UV) run $(PYTHON) src/models/spatiotemporal.py --config $(TRAIN_CONFIG)

eval:
	$(UV) run $(PYTHON) src/utils/metrics.py --checkpoint $(CHECKPOINT)

test:
	$(UV) run pytest $(TEST_PATH)

clean:
	$(UV) run $(PYTHON) -c "from pathlib import Path; import shutil; targets=['data/processed','data/splits','outputs/checkpoints','outputs/results','experiments/logs']; [shutil.rmtree(t, ignore_errors=True) for t in targets if Path(t).exists()]; print('clean: removed generated artifacts')"