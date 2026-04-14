# MERLION

**Meta-Reinforcement Learning via Evolution** for multi-objective optimisation across changing tasks and preference weights.

MERLION is a research codebase for training a **population of meta-policies** rather than a single shared initialisation. It combines:

- **meta-reinforcement learning** for fast adaptation across tasks,
- **multi-objective optimisation** through scalarisation weights,
- **evolutionary search** over preference weights to improve diversity and coverage.

The repository currently contains:

- a general MERLION training implementation under `algorithms/`,
- supply-chain environments under `supply_chain/`,
- toy-problem experiments under `toy_problems/`.

---

## What is in this repository

```text
merlion/
├── algorithms/
│   ├── merlion_main_general.py
│   ├── merlion_finetuning_general.py
│   ├── merlion_archive.py
│   └── merlion_utils.py
├── supply_chain/
│   ├── data_input/
│   ├── fine_tuning/
│   ├── random_env/
│   ├── state_generator/
│   └── training/
├── toy_problems/
│   ├── benchmarks/
│   ├── merlion_env_combine.py
│   ├── merlion_env_combine_notrandom.py
│   ├── run_merlion_general.py
│   └── run_merlion_fine_tuning.py
├── requirements.txt
├── README.md
└── LICENSE
```

### Core files

- `algorithms/merlion_main_general.py`  
  Main MERLION training implementation based on RLlib.

- `algorithms/merlion_finetuning_general.py`  
  Fine-tuning pipeline that restores meta-policies and performs local weight perturbation around each learned preference vector.

- `algorithms/merlion_archive.py`  
  Population archive, objective bookkeeping, and evolutionary update logic.

- `algorithms/merlion_utils.py`  
  Weight initialisation, simplex projection, diversity utilities, and helper functions.

### Supply-chain modules

The supply-chain side of the project is organised around three benchmark families:

- **Simple**
- **Moderate**
- **Complex**

Each family typically includes:

- a **state generator**,
- a **training environment**,
- a **fine-tuning/test environment**.

The environments expose vector rewards for:

1. profit,
2. emissions,
3. service-level inequality.

### Toy-problem modules

The `toy_problems/` folder contains extensions and runners for non-supply-chain benchmarks, useful for cross-domain evaluation of MERLION.

---

## Method overview

At a high level, MERLION works in three phases:

1. **Meta-training**  
   Multiple meta-policies are trained across sampled tasks and scalarisation weights.

2. **Evolutionary update**  
   The associated weight vectors are evolved using fitness signals that balance convergence and diversity.

3. **Fine-tuning**  
   Each meta-policy is adapted to local perturbations of its learned weight vector to obtain a richer Pareto-set approximation on a new task.

This makes MERLION especially suitable for settings where:

- tasks change over time,
- multiple conflicting objectives must be balanced,
- a single shared meta-policy is too restrictive for diverse adaptation.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/rifnyrachman/merlion.git
cd merlion
```

### 2. Create a Python environment

The current repository README and codebase target **Python 3.10.18**.

Using Conda:

```bash
conda create -n merlion_env python=3.10.18
conda activate merlion_env
```

Using `venv`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

For the closest reproduction of the current codebase:

```bash
pip install -r requirements.txt
```

> `requirements.txt` is a broad, pinned research environment and may include packages that are not strictly required for every experiment. It is the safest option for reproducing the repository as it currently stands.

### 4. Verify the key packages

The main stack used by MERLION includes:

- `ray[rllib]` / `ray`
- `gymnasium`
- `torch`
- `numpy`
- `pandas`
- `openpyxl`
- `mo-gymnasium`
- `stable-baselines3` (for some fine-tuning workflows)

You can sanity-check the core setup with:

```bash
python - <<'PY'
import ray
import gymnasium
import torch
import numpy
import pandas
print('Environment looks OK')
PY
```

---

## External dependency: Messiah

### Important note

The current supply-chain implementation depends on **Messiah** for the state representation and operational simulator logic.

In particular, the supply-chain environments and state generators import components such as:

- `messiah.generators.base.Generator`
- `messiah.state.State`
- `messiah.ops`
- `messiah.agents.processing`

If you do **not** have access to Messiah, the supply-chain code will not run as-is.

### License / access caveat

Messiah may require a separate licence or private access, depending on your environment and source setup.

### Open alternative

If Messiah is not available, it can be replaced with an open-source simulator such as **OR-Gym (Hubbs et al, 2020)**: https://arxiv.org/abs/2008.06319.

In that case, the recommended approach is to:

1. replace the current supply-chain state generator with an OR-Gym-based environment,
2. keep the Gymnasium-compatible API,
3. preserve the vector-reward interface,
4. preserve the scalarisation hook:

```python
set_scalarization_weights(w)
```

A replacement environment should ideally expose:

- `reset()` and `step()` following Gymnasium conventions,
- a vector reward stored in `info["mo_reward"]`,
- raw objective values in `info["mo_reward_raw"]`,
- a method for setting scalarisation weights.

That makes it much easier to reuse the MERLION training and fine-tuning code without changing the algorithmic core.

---

## Supply-chain data

The supply-chain benchmarks are driven by spreadsheet-based inputs such as `data_input_v2.xlsx` and related parameter sheets.

These inputs define values such as:

- edge costs,
- node costs,
- emissions,
- initial inventories,
- demand profiles,
- topology-dependent parameters.

If you create a custom benchmark, keep the data format aligned with the corresponding state generator.

---

## How the supply-chain environments are structured

The supply-chain environments are built as Gymnasium environments with:

- continuous action spaces,
- normalised observations,
- three-objective vector rewards,
- scalar rewards computed as the dot product between the reward vector and the active weight vector.

They also wrap the base environment with a lightweight proxy so the trainer can reliably push weights through wrapper stacks.

### Available benchmark scales

- `SimpleSC`
- `ModerateSC`
- `ComplexSC`

### Fine-tuning / evaluation variants

- `TestSimpleSC`
- `TestModerateSC`
- `TestComplexSC`

These are typically used for adaptation or controlled evaluation after meta-training.

---

## Running MERLION

## 1. Meta-training

The main training implementation lives in:

```bash
algorithms/merlion_main_general.py
```

This file defines the MERLION algorithm, configuration, callbacks, worker/task handling, and evolutionary archive integration.

A typical workflow is:

1. register the target environment,
2. build the MERLION config,
3. launch training,
4. save checkpoints for each learned meta-policy.

Because training scripts can vary across experiments, treat `merlion_main_general.py` as the **primary implementation file**, and your project-specific launcher or notebook as the **entry script**.

### Conceptual training flow

```python
from algorithms.merlion_main_general import MERLIONConfig

config = (
    MERLIONConfig()
    .environment(env="YourEnvName")
    .training(
        meta_batch_size=2,
        inner_adaptation_steps=1,
        maml_optimizer_steps=5,
        population_size=10,
    )
)

algo = config.build()
for _ in range(num_iterations):
    result = algo.train()
```

### Key training settings in the current implementation

- framework: **PyTorch**
- `num_rollout_workers = 2`
- `rollout_fragment_length = 200`
- `batch_mode = "complete_episodes"`
- `create_env_on_local_worker = True`
- default `population_size = 10`

---

## 2. Fine-tuning

The general fine-tuning pipeline lives in:

```bash
algorithms/merlion_finetuning_general.py
```

It restores previously saved MERLION meta-policies, applies **local perturbations** around each learned weight vector, and fine-tunes the resulting policies.

### Expected checkpoint layout

The fine-tuning code expects a population directory containing subfolders like:

```text
checkpoint_policy_0/
checkpoint_policy_1/
...
checkpoint_policy_P-1/
```

Each policy folder should contain:

- a `checkpoint_000000/` directory,
- a `weight.json` sidecar file when available.

### Fine-tuning outputs

The fine-tuning pipeline writes:

- per-offspring checkpoints,
- `weight.json`,
- `learning_curve.csv`,
- `all_offspring_summary.csv`,
- `all_timesteps_long.csv`.

### Conceptual fine-tuning call

```python
from algorithms.merlion_finetuning_general import finetune_with_local_perturbations

finetune_with_local_perturbations(
    pop_dir="path/to/meta_policies",
    env_id="YourEnvName",
    out_dir="path/to/output",
    total_steps=5000,
    record_every=500,
    eval_episodes=20,
    m_perturb=5,
    eps=0.05,
    seed=7,
)
```

---

## Example workflow

### Supply-chain workflow

1. Prepare the Python environment.
2. Ensure the supply-chain simulator dependency is available:
   - **Messiah** if reproducing the current implementation, or
   - **OR-Gym / another open-source substitute** if reimplementing the simulator layer.
3. Make sure the input spreadsheets are in the expected locations.
4. Register one of the benchmark environments:
   - simple,
   - moderate,
   - complex.
5. Run meta-training.
6. Save the learned meta-policy population.
7. Run fine-tuning with local weight perturbation.
8. Analyse `all_offspring_summary.csv` and `all_timesteps_long.csv`.

### Toy-problem workflow

1. Set up the Python environment.
2. Use the runners in `toy_problems/`.
3. Train MERLION on the desired benchmark.
4. Fine-tune and compare Pareto-front approximation behaviour.

---

## Reproducibility notes

To improve reproducibility:

- keep Python version fixed,
- use the pinned `requirements.txt`,
- fix random seeds where possible,
- keep the environment registration names and checkpoint layout stable,
- avoid changing reward scaling between training and fine-tuning.

---

## Adapting MERLION to a new environment

To plug a new benchmark into MERLION, the easiest route is to provide a Gymnasium environment with:

- a valid `observation_space`,
- a valid `action_space`,
- `reset()` and `step()`,
- vector reward information in the returned `info`,
- a `set_scalarization_weights(w)` method.

A minimal compatibility target is:

```python
info = {
    "mo_reward": np.array([...], dtype=np.float32),
    "mo_reward_raw": np.array([...], dtype=np.float32),
    "weights": current_weight_vector,
}
```

This is the cleanest way to reuse the existing MERLION callbacks and fine-tuning utilities.

---

## Current README status

The current repository README is intentionally brief. This draft expands it into a fuller project-facing document with:

- clearer repository structure,
- installation steps,
- dependency notes,
- usage guidance,
- supply-chain dependency caveats,
- fine-tuning output expectations.

---

## Citation

If you use this repository in academic work, please cite the corresponding MERLION paper and link back to this repository [to add paper link].

---

## License

This repository is currently distributed under the MIT License. Please also check the licence terms of any external simulator or dataset dependency you plug into the workflow.
