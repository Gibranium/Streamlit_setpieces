import streamlit as st
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mplsoccer import VerticalPitch
import os

# ──────────────────────────────────────────────
# Design tokens — taken from gibranium.github.io/style.css
# Tokens: concrete gray paper, ink, shot-map vermilion
# Charts: afmhot_r gradient (truncated so the light end
# stays visible on the concrete background)
# ──────────────────────────────────────────────
BG = '#D7D1CF'            # concrete gray paper
INK = '#15130F'           # ink
ACCENT = '#C8401F'        # shot-map vermilion
ACCENT_DARK = '#A23217'
MUTED = '#59524D'
GRID = '#ACA7A5'
ON_DARK = '#E9E4E0'
ON_DARK_MUTED = '#9C948D'

AFM = plt.get_cmap('afmhot_r')
AFM_LO, AFM_HI = 0.22, 0.92  # truncation window

def afm_ramp(n, hi_first=True):
    """n colors from the truncated afmhot_r ramp. hi_first=True -> darkest first."""
    if n <= 1:
        return [AFM(AFM_HI)]
    span = np.linspace(AFM_HI, AFM_LO, n) if hi_first else np.linspace(AFM_LO, AFM_HI, n)
    return [AFM(v) for v in span]

def afm_at(frac):
    """Color at a 0-1 fraction of the truncated ramp (1 = darkest)."""
    return AFM(AFM_LO + frac * (AFM_HI - AFM_LO))

def contrast_on(rgba):
    """Ink on light fills, paper on dark fills."""
    r, g, b = rgba[0], rgba[1], rgba[2]
    return INK if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else ON_DARK

# Heatmap: concrete -> afmhot_r ramp, so empty bins melt into the pitch
HEAT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'concrete_afm', [BG] + [mcolors.to_hex(AFM(v)) for v in np.linspace(0.30, 0.95, 6)]
)

# ──────────────────────────────────────────────
# Page config (must be the first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title='Throw-in analysis — Davide Gualano',
    page_icon='🎯',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ──────────────────────────────────────────────
# Fonts for matplotlib (ttf files in repo root)
# ──────────────────────────────────────────────
class FontFallback:
    """Fallback font object when custom fonts can't be loaded"""
    def __init__(self, name='sans-serif'):
        self.name = name

try:
    font_paths = [
        'SourceSansPro-Regular.ttf',
        './SourceSansPro-Regular.ttf',
        '/Users/davidegualano/Documents/Python FTBLData/SourceSansPro-Regular.ttf',
    ]
    font_paths_semibold = [
        'SourceSansPro-SemiBold.ttf',
        './SourceSansPro-SemiBold.ttf',
        'SourceSansPro-Semibold.ttf',
        '/Users/davidegualano/Documents/Python FTBLData/SourceSansPro-SemiBold.ttf',
    ]
    # Display face for figure titles — add Archivo-Bold.ttf to the repo to
    # match the portfolio headers; falls back to Source Sans semibold.
    font_paths_display = [
        'Archivo-Bold.ttf',
        './Archivo-Bold.ttf',
        'Archivo-Black.ttf',
        './Archivo-Black.ttf',
        # Variable font fallback: renders at its default instance (wght ~600)
        'Archivo-VariableFont_wdth_wght.ttf',
        './Archivo-VariableFont_wdth_wght.ttf',
    ]

    regular_font_loaded = False
    semibold_font_loaded = False
    display_font_loaded = False

    fe_regular = FontFallback()
    fe_semibold = FontFallback()
    fe_display = FontFallback()

    for path in font_paths:
        if os.path.exists(path):
            fe_regular = fm.FontEntry(fname=path, name='SourceSansPro-Regular')
            fm.fontManager.ttflist.insert(0, fe_regular)
            regular_font_loaded = True
            break

    for path in font_paths_semibold:
        if os.path.exists(path):
            fe_semibold = fm.FontEntry(fname=path, name='SourceSansPro-SemiBold')
            fm.fontManager.ttflist.insert(1, fe_semibold)
            semibold_font_loaded = True
            break

    for path in font_paths_display:
        if os.path.exists(path):
            fe_display = fm.FontEntry(fname=path, name='Archivo-Display')
            fm.fontManager.ttflist.insert(2, fe_display)
            display_font_loaded = True
            break

    if not semibold_font_loaded and regular_font_loaded:
        fe_semibold = fe_regular
    if not display_font_loaded:
        fe_display = fe_semibold

    if regular_font_loaded:
        matplotlib.rcParams['font.family'] = fe_regular.name
    else:
        st.warning('⚠️ Custom fonts not found. Using system default fonts.')

except Exception as e:
    fe_regular = FontFallback()
    fe_semibold = FontFallback()
    fe_display = FontFallback()
    st.warning(f'⚠️ Could not load custom fonts: {e}. Using system default fonts.')

# Global matplotlib defaults in brand colors
matplotlib.rcParams.update({
    'text.color': INK,
    'axes.labelcolor': MUTED,
    'axes.edgecolor': GRID,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'figure.facecolor': BG,
    'axes.facecolor': BG,
})

# ──────────────────────────────────────────────
# Page CSS — portfolio look: Archivo display, Source Sans 3 body,
# IBM Plex Mono eyebrows, 1.5px ink rules, sharp corners
# ──────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800;900&family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, .stApp, [data-testid="stMarkdownContainer"] p, label, .stRadio, .stSelectbox, .stMultiSelect {{
    font-family: 'Source Sans 3', sans-serif;
}}
h1, h2, h3 {{
    font-family: 'Archivo', sans-serif !important;
    letter-spacing: -0.015em;
    line-height: 1.08;
}}
h1 {{ font-weight: 900 !important; }}
h1 em, h2 em {{ font-style: normal; color: {ACCENT}; }}

.eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {ACCENT};
    margin: 0 0 4px 0;
}}
.section-title {{
    font-family: 'Archivo', sans-serif;
    font-weight: 800;
    font-size: 26px;
    letter-spacing: -0.015em;
    margin: 0 0 4px 0;
    color: {INK};
}}
.section-sub {{
    color: {MUTED};
    font-size: 15px;
    margin: 0 0 8px 0;
    max-width: 720px;
}}
.section-rule {{
    border-top: 1.5px solid {INK};
    margin: 40px 0 18px 0;
}}
.stat-band {{
    display: flex;
    border: 1.5px solid {INK};
    margin: 18px 0 6px 0;
    flex-wrap: wrap;
}}
.stat {{
    flex: 1 1 0;
    min-width: 140px;
    padding: 12px 18px;
    border-right: 1.5px solid {INK};
}}
.stat:last-child {{ border-right: none; }}
.stat-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {MUTED};
}}
.stat-value {{
    font-family: 'Archivo', sans-serif;
    font-weight: 900;
    font-size: 26px;
    color: {INK};
}}
.side-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {ON_DARK_MUTED};
    margin: 14px 0 2px 0;
}}
.side-logo {{
    font-family: 'Archivo', sans-serif;
    font-weight: 800;
    font-size: 17px;
    color: {ON_DARK};
    margin-bottom: 2px;
}}
.side-logo span {{ color: {ACCENT}; }}
.footer-mono {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.05em;
    color: {MUTED};
    margin-top: 8px;
}}
.footer-mono a {{ color: {MUTED}; }}

/* Sharp corners everywhere — brutalist, like the portfolio */
[data-baseweb="select"] > div,
[data-baseweb="input"],
[data-baseweb="tag"],
[data-testid="stExpander"] details,
div[data-baseweb="popover"] > div,
.stButton > button,
[data-testid="stDataFrame"] {{
    border-radius: 0 !important;
}}
/* Vermilion multiselect tags */
[data-baseweb="tag"] {{
    background-color: {ACCENT} !important;
}}
[data-baseweb="tag"] span {{ color: #fff !important; }}

/* Sidebar fallback for Streamlit versions without [theme.sidebar] */
section[data-testid="stSidebar"] {{
    background-color: {INK};
    border-right: 1.5px solid {INK};
}}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] label {{
    color: {ON_DARK};
}}
[data-testid="stExpander"] details {{ border: 1.5px solid {INK}; }}
[data-testid="stHeader"] {{ background: {BG}; }}
</style>
""", unsafe_allow_html=True)


def section(num, label, title, subtitle=None):
    sub = f'<p class="section-sub">{subtitle}</p>' if subtitle else ''
    st.markdown(
        f'<div class="section-rule"></div>'
        f'<p class="eyebrow">{num} — {label}</p>'
        f'<h2 class="section-title">{title}</h2>{sub}',
        unsafe_allow_html=True,
    )


def side_label(text):
    st.sidebar.markdown(f'<p class="side-label">{text}</p>', unsafe_allow_html=True)


def fig_titles(fig, ax, title, subtitle):
    """Left-aligned Archivo title + muted subtitle, no magic data-coordinate offsets."""
    fig.suptitle(title, x=0.01, y=0.995, ha='left', va='top',
                 fontsize=17, fontweight='bold', fontfamily=fe_display.name, color=INK)
    ax.set_title(subtitle, loc='left', fontsize=10.5, color=MUTED,
                 pad=14, fontfamily=fe_regular.name)


# Function to format season ID into a readable format
def format_season_id(season_id):
    season_id = int(season_id)
    start_year = str(season_id - 1)[-2:]
    end_year = str(season_id)[-2:]
    return f"{start_year}/{end_year}"


# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    set_pieces = pd.read_csv('throwins2526.csv', index_col=0)
    team_table = pd.read_csv('throwinstable2526.csv', index_col=0)
    df_atomic = pd.read_csv('throwinsatomic2526.csv', index_col=0)
    return set_pieces, team_table, df_atomic

set_pieces, team_table, df_atomic = load_data()

# ──────────────────────────────────────────────
# Sidebar — console with all controls
# ──────────────────────────────────────────────
st.sidebar.markdown(
    '<p class="side-logo">Davide Gualano<span>.</span></p>'
    '<p style="font-family:\'IBM Plex Mono\',monospace; font-size:11px; '
    f'letter-spacing:0.08em; text-transform:uppercase; color:{ON_DARK_MUTED}; margin:0;">'
    'Set pieces — 2025/26</p>',
    unsafe_allow_html=True,
)

available_competitions = sorted(set_pieces['competition_id'].unique().tolist())

side_label('Competitions')
selected_competitions = st.sidebar.multiselect(
    'Select Competition(s):',
    options=available_competitions,
    default=['ITA-Serie A'] if 'ITA-Serie A' in available_competitions else [available_competitions[0]],
    label_visibility='collapsed',
)

# Filter data based on selected competitions
if selected_competitions:
    set_pieces = set_pieces[set_pieces['competition_id'].isin(selected_competitions)]

    if len(set_pieces) == 0:
        st.error(f"No data available for the selected competition(s): {', '.join(selected_competitions)}")
        st.stop()

    if 'fotmob_id' in set_pieces.columns:
        unique_fotmob_ids = set_pieces['fotmob_id'].unique()
        if 'fotmob_id' in team_table.columns:
            team_table = team_table[team_table['fotmob_id'].isin(unique_fotmob_ids)]
        else:
            st.warning("'fotmob_id' column not found in team_table. Cannot filter teams by competition.")
    else:
        st.warning("'fotmob_id' column not found in set_pieces. Cannot filter teams by competition.")

    if len(team_table) == 0:
        st.error(f"No teams found for the selected competition(s): {', '.join(selected_competitions)}")
        st.stop()
else:
    st.warning('Please select at least one competition.')
    st.stop()


@st.cache_data(show_spinner='Calculating first contact statistics...')
def process_first_contact_data(competitions_tuple, df_atomic):
    """Process atomic data for first contact analysis. Only recalculates when competitions change."""
    competitions = list(competitions_tuple)
    dfa_atomic = df_atomic[df_atomic['competition_id'].isin(competitions)].copy()

    dfa_atomic['max_vaep_next_5s'] = np.nan
    set_pieces_mask = dfa_atomic['type_name'].isin(['throw_in'])

    for idx in tqdm(dfa_atomic[set_pieces_mask].index, desc='Calculating VAEP'):
        current_time = dfa_atomic.loc[idx, 'time_seconds']
        current_game = dfa_atomic.loc[idx, 'game_id']
        current_period = dfa_atomic.loc[idx, 'period_id']
        current_team = dfa_atomic.loc[idx, 'team_id']

        window = dfa_atomic[
            (dfa_atomic['game_id'] == current_game) &
            (dfa_atomic['period_id'] == current_period) &
            (dfa_atomic['team_id'] == current_team) &
            (dfa_atomic['time_seconds'] > current_time) &
            (dfa_atomic['time_seconds'] <= current_time + 5)
        ]

        if len(window) > 0:
            dfa_atomic.loc[idx, 'max_vaep_next_5s'] = window['vaep_value'].max()

    dfa_atomic['max_vaep_next_5s'] = dfa_atomic['max_vaep_next_5s'].fillna(0)
    dfa_atomic['vaep_difference'] = dfa_atomic['max_vaep_next_5s'] - dfa_atomic['vaep_value']

    dfx = dfa_atomic[(dfa_atomic['type_name'].isin(['throw_in'])) & (dfa_atomic['is_inbox'] == True)]
    df2a = dfx[dfx['next_team_name'] == dfx['team_name']]
    df2 = df2a[df2a['next_type_name'].isin(['receival', 'pass', 'goal', 'shot'])]

    games_played = dfa_atomic.groupby(['team_name'])['game_id'].nunique().reset_index(name='games_played')
    first_contacta = dfx.groupby('team_name').size().reset_index(name='set_pieces')
    first_contactb = df2.groupby('team_name').size().reset_index(name='first_contact_won')
    first_contact0 = first_contacta.merge(first_contactb, on='team_name', how='left').fillna(0)
    first_contact = first_contact0.merge(games_played, on='team_name', how='left').fillna(0)
    first_contact['first_contact_ratio'] = first_contact['first_contact_won'] / first_contact['set_pieces']
    first_contact['set_pieces_per_game'] = first_contact['set_pieces'] / first_contact['games_played']

    return first_contact, dfa_atomic

first_contact_data, dfa_atomic = process_first_contact_data(tuple(selected_competitions), df_atomic)

# Remaining sidebar controls (options depend on filtered data)
numeric_columns = [col for col in team_table.columns
                   if col not in ['team_name', 'games_played', 'fotmob_id']
                   and team_table[col].dtype in ['float64', 'int64']]

side_label('Scatter axes')
x_axis = st.sidebar.selectbox(
    'X-Axis:',
    options=numeric_columns,
    index=numeric_columns.index('xG_per_throw_in') if 'xG_per_throw_in' in numeric_columns else 0,
)
y_axis = st.sidebar.selectbox(
    'Y-Axis:',
    options=numeric_columns,
    index=numeric_columns.index('throw_ins_per_game') if 'throw_ins_per_game' in numeric_columns else 1,
)

side_label('Value creation analysis')
throw_in_filter = st.sidebar.radio(
    'Select analysis type:',
    options=[
        'All final third throw-ins (VAEP)',
        'Final third throw-ins excluding box (VAEP)',
        'Possession duration after final third throw-ins',
    ],
    index=0,
    label_visibility='collapsed',
)

side_label('Team maps')
available_teams = sorted(team_table['team_name'].unique().tolist())
selected_team = st.sidebar.selectbox('Select Team:', options=available_teams, index=0,
                                     label_visibility='collapsed') if available_teams else None

st.sidebar.markdown(
    f'<p style="font-family:\'IBM Plex Mono\',monospace; font-size:10px; color:{ON_DARK_MUTED}; '
    'margin-top:28px; line-height:1.7;">'
    '<a href="https://gibranium.github.io" style="color:#9C948D;">Portfolio ↗</a><br>'
    '<a href="https://the-cutback.beehiiv.com" style="color:#9C948D;">The Cutback ↗</a></p>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Main column — header + stat band
# ──────────────────────────────────────────────
competitions_display = ', '.join(selected_competitions)
formatted_season_display = (set_pieces['formatted_season'].iloc[0]
                            if 'formatted_season' in set_pieces.columns and len(set_pieces) > 0 else '')
season_text = f' {formatted_season_display}' if formatted_season_display else ''

st.markdown('<p class="eyebrow">Set pieces — 2025/26</p>', unsafe_allow_html=True)
st.markdown('# Throw-ins, measured. <em>Live.</em>', unsafe_allow_html=True)
st.markdown(
    f'<p class="section-sub">The research pipeline behind '
    f'<a href="https://the-cutback.beehiiv.com" style="color:{ACCENT};">The Cutback</a>\'s '
    f'set-piece coverage, made queryable. Showing <b>{competitions_display}</b>.</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="stat-band">'
    f'<div class="stat"><div class="stat-label">Teams</div><div class="stat-value">{len(team_table)}</div></div>'
    f'<div class="stat"><div class="stat-label">Throw-ins analyzed</div><div class="stat-value">{team_table["total_throw_ins"].sum():,.0f}</div></div>'
    f'<div class="stat"><div class="stat-label">Shots after throw-ins</div><div class="stat-value">{team_table["shots_after_throw_ins"].sum():,.0f}</div></div>'
    '</div>',
    unsafe_allow_html=True,
)

with st.expander(f'Full team table — {len(team_table)} teams'):
    st.dataframe(
        team_table.sort_values(by='xG_per_throw_in', ascending=False).reset_index(drop=True),
        use_container_width=True,
    )

# ──────────────────────────────────────────────
# 01 — Explore: scatter with selectable metrics
# ──────────────────────────────────────────────
x_label = x_axis.replace('_', ' ').title()
y_label_scatter = y_axis.replace('_', ' ').title()

section('01', 'Explore', f'{y_label_scatter} vs {x_label}',
        'Pick the metrics from the sidebar. Color follows the x-axis value.')

fig = plt.figure(figsize=(12, 8), dpi=100, facecolor=BG)
ax = plt.subplot(111, facecolor=BG)

ax.spines['top'].set(visible=False)
ax.spines['right'].set(visible=False)
ax.spines['bottom'].set_color(GRID)
ax.spines['left'].set_color(GRID)

ax.grid(lw=0.1, color=GRID, axis='x', ls='-')
ax.grid(lw=0.1, color=GRID, axis='y', ls='-')

# afmhot_r gradient mapped onto the x-axis metric
x_vals = team_table[x_axis]
if x_vals.max() > x_vals.min():
    fracs = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min())
else:
    fracs = pd.Series(0.5, index=x_vals.index)
point_colors = [afm_at(f) for f in fracs]

ax.scatter(team_table[x_axis], team_table[y_axis], zorder=3, s=90,
           c=point_colors, ec=INK, alpha=0.9, lw=0.6)

x_threshold = team_table[x_axis].quantile(0.75)
y_threshold = team_table[y_axis].quantile(0.75)
outliers = team_table[(team_table[x_axis] >= x_threshold) |
                      (team_table[y_axis] >= y_threshold)]

for idx, row in outliers.iterrows():
    ax.annotate(row['team_name'],
                xy=(row[x_axis], row[y_axis]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, color=INK,
                bbox=dict(boxstyle='round,pad=0.3', fc=ON_DARK, ec=GRID, alpha=0.8, lw=0.5))

ax.set_xlabel(x_label, fontsize=12, color=MUTED, fontfamily=fe_regular.name)
ax.set_ylabel(y_label_scatter, fontsize=12, color=MUTED, fontfamily=fe_regular.name)
ax.set_title(f'{y_label_scatter} vs {x_label} | {competitions_display}{season_text}',
             fontsize=13, color=INK, pad=15, fontfamily=fe_display.name, weight='bold', loc='left')
ax.tick_params(axis='both', labelsize=10, color=GRID, labelcolor=MUTED)

plt.tight_layout()
st.pyplot(fig)

# ──────────────────────────────────────────────
# 02 — First contact won on throw-ins into box
# ──────────────────────────────────────────────
section('02', 'First contact', 'Who wins the box?',
        'Ratio of first contacts won on throw-ins into the box. '
        'Top 20 teams by volume of throw-ins into the box per game — '
        'the circled number on the right.')

if len(first_contact_data) == 0:
    st.warning('No throw-in data available for first contact analysis.')
else:
    first_contact_by_volume = first_contact_data.sort_values(by='set_pieces_per_game', ascending=False).head(20)
    plot_data_fc = first_contact_by_volume.sort_values(by='first_contact_ratio', ascending=False)

    fig_fc, ax_fc = plt.subplots(figsize=(12, 10))
    fig_fc.patch.set_facecolor(BG)
    ax_fc.set_facecolor(BG)

    n_fc = len(plot_data_fc)
    bar_colors = afm_ramp(n_fc, hi_first=True)  # best (top) = darkest

    y_pos = np.arange(n_fc)
    ax_fc.barh(y_pos, plot_data_fc['first_contact_ratio'], color=bar_colors,
               edgecolor=INK, linewidth=0.8, zorder=2)

    ax_fc.set_yticks(y_pos)
    ax_fc.set_yticklabels(plot_data_fc['team_name'], fontsize=11, fontfamily=fe_regular.name)
    ax_fc.invert_yaxis()
    ax_fc.grid(lw=1, color=GRID, axis='x', ls='--')

    ax_fc.set_xlim(0, plot_data_fc['first_contact_ratio'].max() * 1.15)
    ax_fc.set_xlabel('')

    ax_fc.spines['top'].set_visible(False)
    ax_fc.spines['right'].set_visible(False)
    ax_fc.spines['left'].set_visible(False)

    circle_x = plot_data_fc['first_contact_ratio'].max() * 1.1
    for i, (idx, row) in enumerate(plot_data_fc.iterrows()):
        ax_fc.text(circle_x, i, f"{row['set_pieces_per_game']:.1f}",
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   fontfamily=fe_regular.name, color=INK,
                   bbox=dict(boxstyle='circle,pad=0.3', facecolor=BG,
                             edgecolor=INK, linewidth=1.5))

    ax_fc.text(0.956, 1.015, 'Throw-ins into the box\nper game', ha='center', va='bottom',
               fontsize=9, color=MUTED, fontfamily=fe_regular.name, transform=ax_fc.transAxes)

    fig_titles(fig_fc, ax_fc,
               'Ratio of first contacts won on throw ins into box',
               f'Top 20 teams for volume of throw ins into box per game | {competitions_display}{season_text}')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    st.pyplot(fig_fc)

# ──────────────────────────────────────────────
# 03 — Value creation in the final third
# ──────────────────────────────────────────────
section('03', 'Value creation', 'Throw-in value creation in the final third',
        'Distribution showing danger creation or possession duration after throw-ins. '
        'Switch the analysis type from the sidebar.')

if throw_in_filter == 'Possession duration after final third throw-ins':
    df1 = set_pieces[set_pieces['start_x_a0'] >= 75]

    if 'possession_chain_duration' not in df1.columns:
        st.error("'possession_chain_duration' column not found in the dataset.")
        df1 = pd.DataFrame()
    else:
        title_text = 'How long do teams keep possession after a final third throw in?'
        subtitle_text = 'Distribution of length of possession in seconds after throw in.'
        y_label = 'Possession Duration (seconds)'
        metric_column = 'possession_chain_duration'
else:
    df1a = dfa_atomic[dfa_atomic['type_name'] == 'throw_in']
    df1b = df1a[df1a['x_a0'] >= 75].dropna()

    if throw_in_filter == 'All final third throw-ins (VAEP)':
        df1 = df1b
        title_text = 'How much danger do teams create in the 5 seconds after a throw in in the final third?'
    else:
        df1 = df1b[df1b['is_inbox'] != True]
        title_text = 'How much danger do teams create after a final third throw in (excluding those into the box)?'

    subtitle_text = ('Distribution of difference in Atomic VAEP value between throw in and the highest value '
                     'in the following actions in a 5 seconds window.')
    y_label = 'VAEP Difference'
    metric_column = 'vaep_difference'

if len(df1) == 0:
    st.warning('No throw-in data available for the selected analysis.')
else:
    team_medians = df1.groupby('team_name')[metric_column].median().sort_values(ascending=False)
    teams = team_medians.nlargest(20).index.tolist()

    data_to_plot = []
    labels = []
    counts = []

    for team in teams:
        team_data = df1[df1['team_name'] == team][metric_column].dropna()
        if len(team_data) > 0:
            data_to_plot.append(team_data.values)
            labels.append(team)
            counts.append(len(team_data))

    if len(data_to_plot) == 0:
        st.warning('No valid data to display.')
    else:
        fig_vaep, ax_vaep = plt.subplots(figsize=(16, 8))
        fig_vaep.patch.set_facecolor(BG)
        ax_vaep.set_facecolor(BG)

        bp = ax_vaep.boxplot(data_to_plot,
                             positions=range(len(data_to_plot)),
                             widths=0.6,
                             patch_artist=True,
                             showfliers=False,
                             showcaps=False,
                             whiskerprops=dict(visible=False),
                             manage_ticks=True)

        # afmhot_r gradient: best median (left) = darkest, fading right
        box_colors = afm_ramp(len(bp['boxes']), hi_first=True)

        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(INK)
            patch.set_linewidth(1.5)

        for median, color in zip(bp['medians'], box_colors):
            median.set_color(contrast_on(color))
            median.set_linewidth(2)

        all_q1 = [np.percentile(data, 25) for data in data_to_plot]
        all_q3 = [np.percentile(data, 75) for data in data_to_plot]
        y_min = min(all_q1)
        y_max = max(all_q3)
        y_range = y_max - y_min
        padding = y_range * 0.15
        ax_vaep.set_ylim(y_min - padding, y_max + padding)

        ax_vaep.set_xticks(range(len(labels)))
        ax_vaep.set_xticklabels([f'{label}\n({count})' for label, count in zip(labels, counts)],
                                rotation=45, ha='right', fontsize=10, fontfamily=fe_regular.name)
        ax_vaep.set_ylabel(y_label, fontsize=12, fontweight='bold',
                           color=MUTED, fontfamily=fe_regular.name)

        fig_titles(fig_vaep, ax_vaep, title_text,
                   f'{subtitle_text}\nUpper and lower limits of box represent upper and lower quartiles. | '
                   f'Outliers removed from visualisation. {competitions_display}{season_text}')

        ax_vaep.axhline(y=0, color=MUTED, linestyle='--', alpha=0.5, linewidth=1)
        ax_vaep.grid(axis='y', alpha=0.3, color=GRID, linestyle='-', linewidth=0.5)
        ax_vaep.set_axisbelow(True)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        st.pyplot(fig_vaep)

# ──────────────────────────────────────────────
# 04 — Team-specific throw-in maps
# ──────────────────────────────────────────────
section('04', 'Team maps', 'Final third throw-in patterns',
        'Pick a team from the sidebar to map their final third throw-ins, split by pitch side. '
        'Heatmap shows ending coordinates.')

if not available_teams:
    st.warning('No teams available for the selected competition(s).')
elif selected_team:
    setp0 = set_pieces[set_pieces['team_name'] == selected_team]
    setp1 = setp0[setp0['type_name'].isin(['throw_in'])]
    setp2 = setp1[setp1['start_x_a0'] >= 75]
    setpa = setp2[setp2['start_y_a0'] >= 34]
    setpb = setp2[setp2['start_y_a0'] < 34]

    if len(setp2) == 0:
        st.warning(f'{selected_team} has no throw-ins in the final third for the selected competition(s).')
    else:
        fig2 = plt.figure(figsize=(20, 12), constrained_layout=True, facecolor=BG)
        gs = fig2.add_gridspec(3, 6, wspace=0.1, hspace=0.1)

        ax1 = fig2.add_subplot(gs[:, :3])
        ax2 = fig2.add_subplot(gs[0:3, 3:])

        pitch1 = VerticalPitch(pitch_type='custom', pitch_width=68, pitch_length=105, half=True,
                               pad_top=0.4, goal_type='box', pad_bottom=0.4,
                               linewidth=1.25, line_color=INK, pitch_color=BG)
        pitch2 = VerticalPitch(pitch_type='custom', pitch_width=68, pitch_length=105, half=True,
                               pad_top=0.4, goal_type='box', pad_bottom=0.4,
                               linewidth=1.25, line_color=INK, pitch_color=BG)

        ax1.set_facecolor(BG)
        ax2.set_facecolor(BG)

        pitch1.draw(ax=ax1)
        pitch2.draw(ax=ax2)

        ax1.axhline(y=75, color=INK, linestyle='--', linewidth=2, alpha=0.7, zorder=2)
        ax2.axhline(y=75, color=INK, linestyle='--', linewidth=2, alpha=0.7, zorder=2)

        bins = (18, 12)

        if len(setpa) > 0:
            bs_heatmap = pitch1.bin_statistic(setpa.end_x_a0, setpa.end_y_a0, statistic='count', bins=bins)
            pitch1.heatmap(bs_heatmap, ax=ax1, cmap=HEAT_CMAP, zorder=0, alpha=0.7)
            pitch1.arrows(setpa.start_x_a0, setpa.start_y_a0, setpa.end_x_a0, setpa.end_y_a0,
                          width=1.5, zorder=1, alpha=0.5,
                          ec=ON_DARK, fc=INK, headwidth=10, headlength=8, ax=ax1)
            pitch1.scatter(setpa.start_x_a0, setpa.start_y_a0, c=INK, marker='o', s=100, ax=ax1, zorder=1)

        if len(setpb) > 0:
            bs_heatmapy = pitch2.bin_statistic(setpb.end_x_a0, setpb.end_y_a0, statistic='count', bins=bins)
            pitch2.heatmap(bs_heatmapy, ax=ax2, cmap=HEAT_CMAP, zorder=0, alpha=0.7)
            pitch2.arrows(setpb.start_x_a0, setpb.start_y_a0, setpb.end_x_a0, setpb.end_y_a0,
                          width=1.5, zorder=1, alpha=0.5,
                          ec=ON_DARK, fc=INK, headwidth=10, headlength=8, ax=ax2)
            pitch2.scatter(setpb.start_x_a0, setpb.start_y_a0, c=INK, marker='o', s=100, ax=ax2, zorder=1)

        competition_ids = ', '.join(setp1['competition_id'].unique())
        formatted_season = (setp1['formatted_season'].iloc[0]
                            if 'formatted_season' in setp1.columns and len(setp1) > 0 else '')
        season_display = f' {formatted_season}' if formatted_season else ''

        ax1.text(0.5, 1.05, f'Number of throw ins: {setpa.shape[0]}',
                 color=INK, va='center', ha='center', fontsize=11, transform=ax1.transAxes,
                 fontfamily=fe_regular.name)
        ax2.text(0.5, 1.05, f'Number of throw ins: {setpb.shape[0]}',
                 color=INK, va='center', ha='center', fontsize=11, transform=ax2.transAxes,
                 fontfamily=fe_regular.name)

        fig2.text(0.07, 0.905, f'{selected_team} final third throw ins map',
                  fontsize=30, va='center', ha='left', color=INK, fontfamily=fe_display.name)
        fig2.text(0.07, 0.87, f'Heatmap: Ending Coordinates | {competition_ids}{season_display}',
                  fontsize=20, va='center', ha='left', color=MUTED, fontfamily=fe_regular.name)
        fig2.text(0, 0.15, 'X: @gualanodavide | Bluesky: @gualanodavide.bsky.social | '
                  'Linkedin: www.linkedin.com/in/davide-gualano-a2454b187 | Newsletter: the-cutback.beehiiv.com',
                  va='center', ha='left', fontsize=12, color=MUTED, fontfamily=fe_regular.name)

        st.pyplot(fig2)

# ──────────────────────────────────────────────
# 05 — Player throw-in statistics
# ──────────────────────────────────────────────
section('05', 'Players', 'Quarterbacks wannabe',
        'Top 15 players by upper quartile throwing length (minimum 10 throw-ins). '
        'Outlined bar = maximum length, filled bar = upper quartile, circle = average.')

player_data = set_pieces.copy()

player_set_piecesa = player_data.groupby('player_name').size().reset_index(name='total_throw_ins')
player_set_piecesb = player_data.groupby('player_name')['length'].quantile(0.75).reset_index(name='upper_quartile_throwing_length')
player_set_piecesc = player_data.groupby('player_name')['length'].max().reset_index(name='maximum_throwing_length')
player_set_piecesd = player_data.groupby('player_name')['length'].mean().reset_index(name='average_throwing_length')
player_set_pieces0 = player_set_piecesa.merge(player_set_piecesb, on='player_name', how='left').fillna(0)
player_set_pieces1 = player_set_pieces0.merge(player_set_piecesc, on='player_name', how='left').fillna(0)
player_set_pieces = player_set_pieces1.merge(player_set_piecesd, on='player_name', how='left').fillna(0)

player_set_pieces = player_set_pieces[player_set_pieces['total_throw_ins'] >= 10]

if len(player_set_pieces) == 0:
    st.warning('No players have 10 or more throw-ins in the selected competition(s).')
else:
    plot_data = player_set_pieces.sort_values(by='upper_quartile_throwing_length', ascending=False).head(15)

    fig3, ax3 = plt.subplots(figsize=(12, 8))
    fig3.patch.set_facecolor(BG)
    ax3.set_facecolor(BG)

    n_pl = len(plot_data)
    bar_colors = afm_ramp(n_pl, hi_first=True)

    y_pos = np.arange(n_pl)
    ax3.barh(y_pos, plot_data['maximum_throwing_length'], color=[0, 0, 0, 0],
             edgecolor=INK, linewidth=0.8, zorder=2)
    ax3.barh(y_pos, plot_data['upper_quartile_throwing_length'],
             color=bar_colors, edgecolor=INK, linewidth=0.8, zorder=2)

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(plot_data['player_name'], fontsize=11, fontfamily=fe_regular.name)
    ax3.invert_yaxis()
    ax3.grid(lw=1, color=GRID, axis='x', ls='--')

    ax3.set_xlim(0, plot_data['maximum_throwing_length'].max() * 1.15)
    ax3.set_xlabel('')

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    circle_x = plot_data['maximum_throwing_length'].max() * 1.1
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        ax3.text(circle_x, i, f"{row['average_throwing_length']:.2f}",
                 ha='center', va='center', fontsize=9, fontweight='bold',
                 fontfamily=fe_regular.name, color=INK,
                 bbox=dict(boxstyle='circle,pad=0.3', facecolor=BG,
                           edgecolor=INK, linewidth=1.5))

    ax3.text(0.956, 1.015, 'Avg. throwing length (m)', ha='center', va='bottom',
             fontsize=9, color=MUTED, fontfamily=fe_regular.name, transform=ax3.transAxes)

    fig_titles(fig3, ax3, 'Quarterbacks wannabe',
               f'Top 15 players for upper quartile of throw in length | minimum 10 throw ins | '
               f'{competitions_display}{season_text}')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    st.pyplot(fig3)

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown(
    '<div class="section-rule"></div>'
    '<p class="footer-mono">Davide Gualano — '
    '<a href="https://x.com/gualanodavide">X</a> · '
    '<a href="https://bsky.app/profile/gualanodavide.bsky.social">Bluesky</a> · '
    '<a href="https://www.linkedin.com/in/davide-gualano-a2454b187">LinkedIn</a> · '
    '<a href="https://the-cutback.beehiiv.com">The Cutback</a> · '
    '<a href="https://gibranium.github.io">Portfolio</a></p>',
    unsafe_allow_html=True,
)
