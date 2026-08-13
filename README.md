# ⚽ Football Data Dashboard — Davide Gualano

Interactive Streamlit app with two sections: **throw-in analysis** for the 2025/26 season, and
**player scouting** built on percentile dashboards across six position groups.

Pick the section from the sidebar; each one has its own filters.

## 🎯 Throw-ins

Team and player analysis of throw-in events, filtered by competition.

1. **Explore** — scatter plot on any two team metrics, with outliers labelled
2. **First contact** — ratio of first contacts won on throw-ins into the box, top 20 teams by volume
3. **Value creation** — distribution of Atomic VAEP gained in the five seconds after a final-third
   throw-in, or of possession duration afterwards
4. **Team maps** — one team's final-third throw-ins, split by pitch side, over a heatmap of ending
   coordinates
5. **Players** — leaderboard on any throwing metric, filtered by team and minimum volume
6. **Distribution** — interquartile spread of throwing length for the ranked players

## 📋 Player scouting

Every player scored by percentile against a pool you define, using the metrics from my evaluation
notebooks. Six position groups: goalkeepers, centre-backs, full-backs and wing-backs, central
midfielders, attacking midfielders and wingers, strikers.

- **Pool filters** (minimum minutes, competitions) decide *what the percentiles are measured
  against*. Move the minutes floor and every number in the table changes, not just which rows pass.
- **Search filters** (name, season, team, role) only narrow which player you look at. The pool is
  unaffected.
- **Query the table** on as many metrics as you like — each gets a 0–100 percentile range, and a
  player has to clear all of them. Results are downloadable as CSV.
- **Dashboard** for any match: paired bars where the shaded bar is quality (percentile-coloured)
  and the hatched bar is volume, grouped into panels by phase of play. Downloadable as PNG.

## 🚀 Live demo

[Link to deployed app will go here]

## 💻 Local installation

### Prerequisites

- Python 3.9+
- Git LFS (for the throw-in CSVs)

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/throwins-visualization.git
cd throwins-visualization
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501`.

## 📁 Files

| File | Purpose |
| --- | --- |
| `app.py` | The Streamlit app — layout, filters, throw-in visualisations |
| `player_dashboard.py` | Percentile dashboard config and renderer |
| `check_app.py` | Smoke test (optional, never runs as part of the app) |

Throw-in data, stored with Git LFS:

- `throwins2526.csv` — individual throw-in events
- `throwinstable2526.csv` — aggregated team statistics
- `throwinsatomic2526.csv` — atomic VAEP data for throw-ins

Scouting data, one file per position group, exported from the `EVALUATION-*.ipynb` notebooks:

- `dashboard_GK.parquet`, `dashboard_CB.parquet`, `dashboard_WB.parquet`,
  `dashboard_CDM.parquet`, `dashboard_AMW.parquet`, `dashboard_ST.parquet`

Parquet rather than CSV because `competition_id` and `season_id` are list-valued; a CSV round-trip
turns them into strings and the filters stop matching. Refreshing the data is just re-running the
export cell and replacing the files — no code change, and the season and competition dropdowns
rebuild themselves.

If the scouting section says the exports are missing, check they aren't caught by `.gitignore`.

## 🛠️ Technologies

- **Streamlit** — web framework
- **pandas** / **NumPy** — data manipulation
- **Matplotlib** — visualisation
- **mplsoccer** — pitch plots
- **pyarrow** — parquet reader

## 📈 Metrics explained

- **xG per throw-in** — expected goals generated per throw-in
- **First contact ratio** — share of throw-ins into the box where the team wins the first contact
- **VAEP** — Valuing Actions by Estimating Probabilities; here, the danger created in the five
  seconds after an action
- **Possession duration** — how long a team keeps the ball after a throw-in
- **PAx100** — passes completed above expectation, per 100 attempts
- **Percentile pool** — the population a player is ranked against; set by the minutes and
  competition filters, not by the search

## 👤 Author

**Davide Gualano** — football data analysis, consulting and recruitment.

- 📧 [davide@davidegualano.com](mailto:davide@davidegualano.com)
- 🌐 [davidegualano.com](https://davidegualano.com) · **[Work with me](https://davidegualano.com/work-with-me.html)**
- [LinkedIn](https://www.linkedin.com/in/davide-gualano-a2454b187) ·
  [X](https://x.com/gualanodavide) ·
  [Bluesky](https://bsky.app/profile/gualanodavide.bsky.social) ·
  [The Cutback](https://the-cutback.beehiiv.com)

## 📄 License

MIT.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.
