# Traffic-signal RL demo UI

A Gradio app to watch and compare Q-learning, SARSA, and DQN controlling
a four-way traffic intersection, based on the Project 6 notebook.

## Files

- `environment.py` -- the traffic intersection environment
- `discretize.py` -- state bucketing for tabular methods
- `agents.py` -- Q-learning, SARSA, DQN, replay buffer
- `train_all.py` -- trains all three agents, saves them to `saved_models/`
- `visualize.py` -- draws the intersection state as a matplotlib figure
- `app.py` -- the Gradio UI (live simulation + comparison dashboard)

## Setup

```bash
cd traffic_rl_ui
pip install -r requirements.txt
```

## 1. Train the agents (run once)

```bash
python train_all.py
```

This trains Q-learning, SARSA, and DQN (2000 episodes each by default --
edit `NUM_EPISODES` in `train_all.py` to change it) and saves everything
into `saved_models/`. This takes a few minutes on a laptop CPU.

## 2. Launch the app

```bash
python app.py
```

Open the printed URL (usually `http://127.0.0.1:7860`) in your browser.

- **Live simulation tab:** pick a controller (Baseline / Q-learning /
  SARSA / DQN), click "New episode," then "Step" repeatedly to watch it
  choose green lights and clear queues.
- **Comparison dashboard tab:** click "Load results" to see bar charts
  of average reward and average queue length across the three trained
  agents.

## Notes

- Retraining is separate from the UI on purpose -- training takes
  minutes, the UI should load instantly. If you retrain with different
  settings, just rerun `train_all.py` before relaunching `app.py`.
- No external API key is needed anywhere in this project -- everything
  runs locally.
