# ⚽ Throw-ins Visualization Dashboard

Interactive Streamlit dashboard for analyzing throw-in statistics from football/soccer matches in the 2025/26 season.

## 🎯 Features

- **Competition Filtering**: Select one or multiple competitions to analyze
- **Team Statistics**: View comprehensive throw-in metrics including xG, shots, and possession data
- **Interactive Scatter Plots**: Choose any metrics for X and Y axes with team logos
- **First Contact Analysis**: Bar charts showing teams' success in winning first contact on throw-ins
- **VAEP Analysis**: Box plots showing value creation after throw-ins
- **Team-Specific Maps**: Detailed pitch visualizations for individual teams' throw-in patterns
- **Player Statistics**: Rankings of players by throwing distance

## 📊 Visualizations Include

1. Customizable scatter plot with team logos
2. First contact ratio on throw-ins into the box
3. VAEP difference distribution (danger creation)
4. Possession duration after throw-ins
5. Team-specific throw-in heatmaps
6. Player throwing distance rankings

## 🚀 Live Demo

[Link to deployed app will go here]

## 💻 Local Installation

### Prerequisites
- Python 3.8+
- Git LFS installed

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/throwins-visualization.git
cd throwins-visualization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

4. Open your browser at `http://localhost:8501`

## 📁 Data Files

The dashboard uses three main data files (stored with Git LFS):
- `throwins2526.csv` - Individual throw-in events
- `throwinstable2526.csv` - Aggregated team statistics
- `throwinsatomic2526.csv` - Atomic VAEP data for throw-ins

## 🛠️ Technologies

- **Streamlit**: Web framework
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **mplsoccer**: Football pitch visualizations
- **NumPy & SciPy**: Statistical analysis

## 📈 Metrics Explained

- **xG per throw-in**: Expected goals generated per throw-in
- **First contact ratio**: Percentage of throw-ins where team wins first contact
- **VAEP**: Valuing Actions by Estimating Probabilities - measures danger created
- **Possession duration**: How long team maintains possession after throw-in

## 👤 Author

**Davide Gualano**

- X: [@gualanodavide](https://twitter.com/gualanodavide)
- Bluesky: [@gualanodavide.bsky.social](https://bsky.app/profile/gualanodavide.bsky.social)
- LinkedIn: [Davide Gualano](https://www.linkedin.com/in/davide-gualano-a2454b187)
- Newsletter: [The Cutback](https://the-cutback.beehiiv.com)

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!
