# 🎮 Ensemble Learning - ML Battle Arena

A gamified Streamlit app for learning **ensemble learning** (Voting, Bagging, Boosting, Stacking). Complete six levels, earn XP,
build answer streaks, unlock 12 achievements, take quizzes and finish with a Final Championship. Every level is powered by **real
scikit-learn models** - nothing is faked.

## What is inside
| Page | What you do |
|---|---|
| 🏠 **Home** | Player card (level, XP, score, streak), mission cards (Completed / In Progress / Locked) |
| 🎯 **Missions** | **L1** Prediction Rookie · **L2** Random Forest Squad · **L3** Boosting Challenge · **L4** Voting Arena · **L5** Stacking Mastership · **L6** Final Championship |
| 🧪 **Algorithm Playground** | Live visualization: Dataset · Decision Boundary · Individual Models · Ensemble · Metrics tabs, 11 algorithms (+ XGBoost if installed) |
| 📚 **Theory Library** | 13 topics with diagrams + interview Q&A |
| 📝 **Quizzes** | 5 random questions from a bank of 18, +50 XP per first-time correct answer |
| 🏆 **Leaderboard** | You vs. demo players (local only) |
| ⭐ **Achievements** | 12 badges |
| ⚙️ **Settings** | Player name, export / import progress (JSON), reset |

### Game rules
- A mission is completed when its **objective** is met *and* its **checkpoint question** is answered correctly. Levels unlock in order.
- Level = completed missions + 1. XP bar goal = 1000 XP. Achievements pay bonus XP.
- Streak = consecutive correct answers (checkpoints, quiz, voting rounds); a wrong answer resets it.
- **Voting Arena:** predict the hard- or soft-vote of three models; 3 correct rounds complete the level.
- **Final Championship:** 6-step wizard (Dataset → Select Models → Configure → Train → Compare → Choose). You win if your pick's test F1 is within 0.005 of the best.
- Progress lives in your browser **session**. Use *Settings → Export progress* to keep it. There is no server-side score storage.
link- https://aloksingh2508.github.io/Ensemble-Areena/

## Project structure
```
ensemble-learning-game/
├── app/
│   ├── main.py              # entry point: theme, header, st.navigation
│   ├── style.py             # dark arena CSS + cards
│   ├── game/                # state.py (XP/streak/achievements), content.py (missions, quiz, theory)
│   ├── missions/            # level1.py ... level6.py, common.py
│   └── pages/               # home, missions, playground, theory, quiz, leaderboard, achievements, settings
├── src/                     # tested ML core: data, models, training, plots, ui, content
├── tests/                   # test_core.py (ML core) + test_game.py (game logic and every mission flow)
├── .streamlit/config.toml   # dark theme
├── requirements.txt  requirements-dev.txt  pytest.ini
└── README.md
```

## Run locally (Windows + VS Code)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app/main.py
pytest
```
macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`.

## Deploy
Push to GitHub, then on https://share.streamlit.io create an app with **Main file path: `app/main.py`**.

## Tech stack
Python · Streamlit · scikit-learn · NumPy · Pandas · Matplotlib · Plotly

## Run as a static website (no Python server)
The whole app can run **inside the browser** with [stlite](https://github.com/whitphx/stlite) (Streamlit + scikit-learn compiled to WebAssembly).
```powershell
python scripts/build_web.py            # creates web/index.html (self-contained)
python -m http.server -d web 8000      # then open http://localhost:8000
```
- Open it through `http://`, not by double-clicking the file (browsers block module workers on `file://`).
- Host the `web/` folder on **GitHub Pages**, **Netlify** (drag & drop), or **Cloudflare Pages** - no server needed.
- The first visit downloads ~60-100 MB (Python + scikit-learn) and takes a minute; it needs internet access to `cdn.jsdelivr.net`. Heavy steps (Stacking, big forests) run slower than on a normal server.
- Re-run `build_web.py` after every code change.
