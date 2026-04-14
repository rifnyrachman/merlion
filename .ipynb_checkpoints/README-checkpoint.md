# merlion
MERLION: a multi-objective meta-reinforcement learning framework for learning diverse policies across preference weights and tasks.

---

## Repository Structure

algorithms/ main script for merlion
supplychain/ script for supply chain problems
toyproblem/ script for toy problems

---

## Environment Setup

This project uses **Python 3.10.18** and dependencies listed in `requirements.txt`.

### 1. Create Conda Environment

```bash
conda create -n merlion_env python=3.10.18

```

run merlion_main_general.py first for meta-training
then run merlion_finetuning_general.py for fine-tuning phase


### 2. Activate Environment
```bash
conda activate merlion_env
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```