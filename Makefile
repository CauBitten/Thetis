.PHONY: help preprocess episodes episodes-6-3-3 train eval test clean

UV ?= uv
PYTHON ?= python

DATASET_DIR ?= dataset
DATA_DIR ?= data
SEED ?= 42

MANIFEST ?= data/processed/manifest.csv
EPISODES_DIR ?= data/episodes/
N_WAY ?= 5
N_WAY_VAL ?= 3
N_WAY_TEST ?= 3
K_SHOT ?= 5
Q_QUERY ?= 15
EPISODES_PER_SPLIT ?= 1000
N_TRAIN ?= 6
N_VAL ?= 3
N_TEST ?= 3

TRAIN_CONFIG ?= experiments/configs/protonet_rgb_5w5s.yaml
CHECKPOINT ?= outputs/checkpoints/latest/best.pt
TEST_PATH ?= tests/

help:
	@echo "Available targets:"
	@echo "  make preprocess        # build manifest, integrity report, and label index"
	@echo "  make episodes          # sample N-way K-shot episodes (uniform n_way across splits)"
	@echo "  make episodes-6-3-3    # sample episodes with 6/3/3 class split + asymmetric n_way (5/3/3)"
	@echo "  make train             # meta-train with TRAIN_CONFIG (default: protonet_rgb_5w5s)"
	@echo "  make eval              # evaluate CHECKPOINT on meta_test"
	@echo "  make test              # run test suite"
	@echo "  make clean             # remove generated artifacts"

preprocess:
	$(UV) run $(PYTHON) src/data/loader.py --input $(DATASET_DIR) --output $(DATA_DIR) --seed $(SEED)

episodes:
	$(UV) run $(PYTHON) src/data/episode_sampler.py \
		--manifest $(MANIFEST) \
		--output $(EPISODES_DIR) \
		--n-way $(N_WAY) --k-shot $(K_SHOT) --q-query $(Q_QUERY) \
		--train-classes $(N_TRAIN) --val-classes $(N_VAL) --test-classes $(N_TEST) \
		--episodes-per-split $(EPISODES_PER_SPLIT) \
		--seed $(SEED)

# Phase-2 default: 6/3/3 partition with 5-way train, 3-way val/test
# (max viable n_way given the 12-class budget — see README "Splits" section).
episodes-6-3-3:
	$(UV) run $(PYTHON) src/data/episode_sampler.py \
		--manifest $(MANIFEST) \
		--output $(EPISODES_DIR) \
		--n-way 5 --n-way-val 3 --n-way-test 3 \
		--k-shot $(K_SHOT) --q-query $(Q_QUERY) \
		--train-classes 6 --val-classes 3 --test-classes 3 \
		--episodes-per-split $(EPISODES_PER_SPLIT) \
		--seed $(SEED)

train:
	$(UV) run $(PYTHON) src/training/meta_trainer.py --config $(TRAIN_CONFIG)

eval:
	$(UV) run $(PYTHON) src/training/eval_episodic.py \
		--checkpoint $(CHECKPOINT) \
		--episodes $(EPISODES_DIR)/meta_test/episodes.jsonl

test:
	$(UV) run pytest $(TEST_PATH)

clean:
	$(UV) run $(PYTHON) -c "from pathlib import Path; import shutil; targets=['data/processed','data/splits','outputs/checkpoints','outputs/results','experiments/logs']; [shutil.rmtree(t, ignore_errors=True) for t in targets if Path(t).exists()]; print('clean: removed generated artifacts')"