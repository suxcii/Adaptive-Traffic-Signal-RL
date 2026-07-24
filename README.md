# Adaptive Traffic Signal RL

A four-way intersection is modeled as a 16-dimensional queue-length MDP
(12 incoming movements + 4 outgoing roads, 12 possible green-light actions).
Three learned controllers — tabular Q-learning, tabular SARSA, and a Deep
Q-Network (DQN) — are trained and compared against a longest-queue-first
baseline. This repo has three parts:

```
Adaptive-Traffic-Signal-RL/
├── training/    Colab notebook — environment design, training, evaluation, plots
├── backend/     FastAPI service — loads the trained models, runs the env, serves decisions
└── frontend/    React/Vite app — animated intersection diagram, talks to the backend
```

**How it fits together:** `training/` is where the research happens (hyperparameter
studies, seeded final evaluation, report figures). `backend/` is a standalone,
already-trained deployment of the same three algorithms — it ships with pretrained
weights in `backend/saved_models/`, so the demo UI never retrains on startup. It only
loads the saved policies and serves a decision per API call. `frontend/` is a pure
display layer: it sends the current traffic state to the backend and renders whatever
action comes back.

## 1. Training (optional — only needed if you want to reproduce or change the models)

```bash
cd training
pip install -r requirements-training.txt
jupyter notebook Adaptive_Traffic_Signal_RL.ipynb
```

Run top to bottom. The canonical, reported results come from the seeded final
evaluation step (`GLOBAL_SEED = 42`) — see the in-notebook notes for details. Full
5,000-episode training runs are the slow part; the DQN sensor-noise sweep is the
most expensive single cell.

This notebook is the source of truth for the RL methodology. The `backend/` copy of
the environment and agents (`environment.py`, `agents.py`) is a lightweight, standalone
port used to serve the live demo quickly — if you change the MDP or reward design in
the notebook, port the change to `backend/environment.py` and `backend/agents.py` too,
then retrain (step 2 below) so the saved models stay in sync.

## 2. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

Pretrained weights already ship in `saved_models/` (`q_table.pkl`, `sarsa_table.pkl`,
`dqn.pt`), so this works immediately with all four controllers, including the
**Baseline** (always serves the largest incoming queue — no training needed).

To retrain from scratch:

```bash
python train_all.py
```

This overwrites `saved_models/` (2,000 episodes per algorithm by default — bump
`NUM_EPISODES` in `train_all.py` for closer-to-final results, matching the 5,000 used
in the notebook). Restart `uvicorn` afterward so it picks up the new files. Selecting
an algorithm that isn't trained yet falls back to Baseline automatically.

Leave this running in one terminal — it serves the API at `http://127.0.0.1:8000`.

## 3. Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the printed URL (usually `http://localhost:5173`). It talks to the backend at
`http://127.0.0.1:8000` — change `API_BASE` at the top of `src/app/App.tsx` if you run
the backend elsewhere.

## What's wired up

- **New Episode** — resets the environment, starts fresh
- **Next Step** — one `env.step()` call; the chosen lane lights up green on the
  diagram and in the side panel, queue counts update
- **Previous** — steps back through already-computed snapshots (no recompute)
- **Algorithm dropdown** — switches which policy picks actions (Baseline /
  Q-learning / SARSA / DQN)
- Incoming lane counts, outgoing (cumulative exited) counts, reward, total queue,
  episode/step number, and cars passed are all live from the backend

## Notes

- The backend keeps one global in-memory episode — it's a local dev/demo tool for
  one person at a time, not a multi-user service.
- No external API key is needed anywhere in this project — everything runs locally.
