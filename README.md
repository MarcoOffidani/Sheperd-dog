# Evacuation RL (convex / non-convex geometries)

Ambiente di reinforcement learning per studiare l'evacuazione di pedoni in stanze con geometrie **convex** e **non-convex**. Il focus e' su dinamiche collettive (leader/followers, viscek-like), reward di evacuazione e parametri ambientali configurabili per esperimenti riproducibili.

## Struttura del progetto

File e cartelle principali:

- `main.py`: entry point per training con Stable-Baselines3 (PPO).
- `run.py`: entry point alternativo (caricamento/salvataggio modelli, wandb).
- `src/env/`: ambiente di simulazione, geometrie, costanti.
- `src/env/constants.py`: parametri di default dell'ambiente.
- `src/utils.py`: parsing CLI e helper (experiment name, wrapper).
- `src/params.py`: parametri di training/logging (tensorboard, modelli, ecc.).
- `requirements.txt`: dipendenze Python.
- `saved_data/`: output (gif, png, modelli, log, tensorboard).
- `wandb/`: log di Weights & Biases (se abilitato).

## Installazione

### Con `venv`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Con `conda`

```bash
conda create -n evacuation-rl python=3.10
conda activate evacuation-rl
pip install -r requirements.txt
```

## Uso da linea di comando

Esempi reali (da root del repo):

```bash
# Training base (PPO) su geometria convex, con gif a ogni step
python main.py --exp-name "baseline_convex" --learn-timesteps 200000 --number-of-pedestrians 50 --draw
```

```bash
# Training con embedding di gravita disabilitato e posizioni relative abilitate
python main.py --exp-name "relpos_no_gravity" --enabled-gravity-embedding false --use-relative-positions true --learn-timesteps 100000
```

```bash
# Variante con parametri di reward e dinamica leader
python main.py --exp-name "reward_variant" --enslaving-degree 0.3 --intrinsic-reward-coef 0.5 --is-new-exiting-reward true --is-new-followers-reward false
```

Note rapide:

- `--draw` salva le traiettorie in gif (utile per ispezione qualitativa, piu lento).
- Per geometrie **non-convex**, i muri e le aperture sono definiti in `src/env/constants.py` (non via CLI).

## CLI configuration (src/env/constants.py)

Di seguito l'elenco esatto dei parametri di `src/env/constants.py` che possono essere configurati via CLI (tramite `src/utils.py`) e di quelli non configurabili.

Configurabili via CLI:

- `NUM_PEDESTRIANS`  (`--number-of-pedestrians`)
- `WIDTH` (`--width`)
- `HEIGHT` (`--height`)
- `STEP_SIZE` (`--step-size`)
- `NOISE_COEF` (`--noise-coef`)
- `NUM_OBS_STACKS` (`--num-obs-stacks`)
- `USE_RELATIVE_POSITIONS` (`--use-relative-positions`)
- `DEACTIVATE_WALLS` (`--deactivate-walls`)
- `ENSLAVING_DEGREE` (`--enslaving-degree`)
- `IS_NEW_EXITING_REWARD` (`--is-new-exiting-reward`)
- `IS_NEW_FOLLOWERS_REWARD` (`--is-new-followers-reward`)
- `INTRINSIC_REWARD_COEF` (`--intrinsic-reward-coef`)
- `TERMINATION_AGENT_WALL_COLLISION` (`--is-termination-agent-wall-collision`)
- `INIT_REWARD_EACH_STEP` (`--init-reward-each-step`)
- `MAX_TIMESTEPS` (`--max-timesteps`)
- `N_EPISODES` (`--n-episodes`)
- `N_TIMESTEPS` (`--n-timesteps`)
- `ENABLED_GRAVITY_EMBEDDING` (`--enabled-gravity-embedding`)
- `ENABLED_GRAVITY_AND_SPEED_EMBEDDING` (`--enabled-gravity-embedding-speed`)
- `ALPHA` (`--alpha`)

Non configurabili via CLI (solo editing diretto del file):

- `EPS`
- `SWITCH_DISTANCE_TO_LEADER`
- `SWITCH_DISTANCE_TO_OTHER_PEDESTRIAN`
- `SWITCH_DISTANCE_TO_EXIT`
- `SWITCH_DISTANCE_TO_ESCAPE`
- `SAVE_PATH_GIFF`
- `SAVE_PATH_PNG`
- `SAVE_PATH_LOGS`
- `WALL_HOLE_HALF_WIDTH`
- `WALLS`
- `WALL2`
- `WALL3`
- `VERTICAL_WALL_POSITION`
- `VERTICAL_WALL_HALF_WIDTH`

## DEACTIVATE_WALLS / --deactivate-walls

Disabilita la logica/disegno/interazioni dei muri interni, mantenendo invariati i confini esterni (collisioni con i bordi della stanza).

Esempio minimo:

```bash
python main.py --exp-name "no_walls" --deactivate-walls
```

## Output

Output generati (default):

- `saved_data/giff/`: gif delle traiettorie
- `saved_data/png/`: png delle traiettorie
- `saved_data/models/`: modelli addestrati
- `saved_data/logs/`: log per esperimenti
- `saved_data/tb-logs/`: TensorBoard
- `wandb/`: log di Weights & Biases (se abilitato)
