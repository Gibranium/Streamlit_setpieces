import streamlit as st
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import urllib.request
from mplsoccer import VerticalPitch
import os

# Load custom fonts for visualization with error handling
class FontFallback:
    """Fallback font object when custom fonts can't be loaded"""
    def __init__(self):
        self.name = 'sans-serif'

try:
    # Try to load custom fonts from multiple possible locations
    font_paths = [
        'SourceSansPro-Regular.ttf',  # In repository root (Streamlit Cloud)
        './SourceSansPro-Regular.ttf',
        '/Users/davidegualano/Documents/Python FTBLData/SourceSansPro-Regular.ttf',  # Local dev
    ]
    font_paths_semibold = [
        'SourceSansPro-SemiBold.ttf',  # In repository root (Streamlit Cloud) - note capital B
        './SourceSansPro-SemiBold.ttf',
        'SourceSansPro-Semibold.ttf',  # Try lowercase 'b' as well
        '/Users/davidegualano/Documents/Python FTBLData/SourceSansPro-SemiBold.ttf',  # Local dev
    ]
    
    regular_font_loaded = False
    semibold_font_loaded = False
    
    # Initialize with fallback
    fe_regular = FontFallback()
    fe_semibold = FontFallback()
    
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
    
    # If semibold didn't load but regular did, use regular for both
    if not semibold_font_loaded and regular_font_loaded:
        fe_semibold = fe_regular
        st.info("ℹ️ Using regular font for both regular and bold text.")
    
    if regular_font_loaded:
        matplotlib.rcParams['font.family'] = fe_regular.name
    else:
        st.warning("⚠️ Custom fonts not found. Using system default fonts.")
        
except Exception as e:
    # Fallback to default font if loading fails
    fe_regular = FontFallback()
    fe_semibold = FontFallback()
    st.warning(f"⚠️ Could not load custom fonts: {e}. Using system default fonts.")

# Function to format season ID into a readable format
def format_season_id(season_id):
    # Convert to integer if it's a float
    season_id = int(season_id)
    # Extract the last two digits of the year
    start_year = str(season_id - 1)[-2:]
    # Calculate the end year
    end_year = str(season_id)[-2:]
    # Format as 20/21
    formatted_season = f"{start_year}/{end_year}"
    return formatted_season

st.title("2025/26 Throw-ins visualizations")
st.subheader("Select a league or more leagues to visualize throw-ins data!")

# Cache the data loading
@st.cache_data
def load_data():
    set_pieces = pd.read_csv('throwins2526.csv', index_col=0)
    team_table = pd.read_csv('throwinstable2526.csv', index_col=0)
    df_atomic = pd.read_csv('throwinsatomic2526.csv', index_col=0)
    return set_pieces, team_table, df_atomic

set_pieces, team_table, df_atomic = load_data()

# Get unique competition IDs for the filter
available_competitions = sorted(set_pieces['competition_id'].unique().tolist())

# Add multiselect filter for competitions
selected_competitions = st.multiselect(
    'Select Competition(s):',
    options=available_competitions,
    default=['ITA-Serie A'] if 'ITA-Serie A' in available_competitions else [available_competitions[0]]
)

# Filter data based on selected competitions
if selected_competitions:
    # First, filter set_pieces by competition
    set_pieces = set_pieces[set_pieces['competition_id'].isin(selected_competitions)]
    
    # Check if filtering resulted in empty dataframes
    if len(set_pieces) == 0:
        st.error(f"No data available for the selected competition(s): {', '.join(selected_competitions)}")
        st.stop()
    
    # Get unique fotmob_ids from filtered set_pieces
    if 'fotmob_id' in set_pieces.columns:
        unique_fotmob_ids = set_pieces['fotmob_id'].unique()
        # Filter team_table to only include teams with these fotmob_ids
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
    st.warning("Please select at least one competition.")
    st.stop()

# Cache the atomic data processing for first contact analysis
@st.cache_data(show_spinner="Calculating first contact statistics...")
def process_first_contact_data(competitions_tuple, df_atomic):
    """Process atomic data for first contact analysis. Only recalculates when competitions change."""
    competitions = list(competitions_tuple)
    dfa_atomic = df_atomic[df_atomic['competition_id'].isin(competitions)].copy()
    
    # Calculate VAEP difference for throw-ins
    # Create a column to store the max vaep_value in the next 5 seconds
    dfa_atomic['max_vaep_next_5s'] = np.nan
    
    # Filter for throw-ins
    set_pieces_mask = dfa_atomic['type_name'].isin(['throw_in'])
    
    for idx in tqdm(dfa_atomic[set_pieces_mask].index, desc="Calculating VAEP"):
        # Get the current row details
        current_time = dfa_atomic.loc[idx, 'time_seconds']
        current_game = dfa_atomic.loc[idx, 'game_id']
        current_period = dfa_atomic.loc[idx, 'period_id']
        current_team = dfa_atomic.loc[idx, 'team_id']
        
        # Find all actions in the next 5 seconds (same game, period, same team)
        window = dfa_atomic[
            (dfa_atomic['game_id'] == current_game) &
            (dfa_atomic['period_id'] == current_period) &
            (dfa_atomic['team_id'] == current_team) &
            (dfa_atomic['time_seconds'] > current_time) &
            (dfa_atomic['time_seconds'] <= current_time + 5)
        ]
        
        # Get the maximum vaep_value in this window
        if len(window) > 0:
            max_vaep = window['vaep_value'].max()
            dfa_atomic.loc[idx, 'max_vaep_next_5s'] = max_vaep
    
    # Fill remaining NaN with 0
    dfa_atomic['max_vaep_next_5s'] = dfa_atomic['max_vaep_next_5s'].fillna(0)
    dfa_atomic['vaep_difference'] = dfa_atomic['max_vaep_next_5s'] - dfa_atomic['vaep_value']
    
    # Filter for throw-ins into the box
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

# Display summary statistics
st.write("### Summary Statistics")
competitions_display = ', '.join(selected_competitions)
st.write(f"**Selected Competition(s):** {competitions_display}")
st.write(f"Total teams: {len(team_table)}")
st.write(f"Total throw-ins analyzed: {team_table['total_throw_ins'].sum():.0f}")
st.write(f"Total shots after throw-ins: {team_table['shots_after_throw_ins'].sum():.0f}")

# Display the data table
st.write("### Team Statistics Table (Filtered by Selected Competition)")
st.write(f"Showing {len(team_table)} teams from: **{competitions_display}**")
st.dataframe(
    team_table.sort_values(by='xG_per_throw_in', ascending=False).reset_index(drop=True),
    use_container_width=True
)

# Get numeric columns for selection (excluding team_name, games_played, and fotmob_id)
numeric_columns = [col for col in team_table.columns if col not in ['team_name', 'games_played', 'fotmob_id'] and team_table[col].dtype in ['float64', 'int64']]

# Add metric selection
st.write("### Select Your Metrics")
col1, col2 = st.columns(2)

with col1:
    x_axis = st.selectbox(
        'X-Axis:',
        options=numeric_columns,
        index=numeric_columns.index('xG_per_throw_in') if 'xG_per_throw_in' in numeric_columns else 0
    )

with col2:
    y_axis = st.selectbox(
        'Y-Axis:',
        options=numeric_columns,
        index=numeric_columns.index('throw_ins_per_game') if 'throw_ins_per_game' in numeric_columns else 1
    )

# Create the figure with specified style
fig = plt.figure(figsize=(12, 8), dpi=100, facecolor='#D7D1CF')
ax = plt.subplot(111, facecolor='#D7D1CF')

# Customization of the spines
ax.spines["top"].set(visible=False)
ax.spines["right"].set(visible=False)
ax.spines["bottom"].set_color('#ACA7A5')
ax.spines["left"].set_color('#ACA7A5')

# Customization of the grid
ax.grid(lw=0.1, color="#ACA7A5", axis='x', ls="-")
ax.grid(lw=0.1, color="#ACA7A5", axis='y', ls="-")

# Plot scatter points with selected axes (s=0 since logos are the visual markers)
ax.scatter(team_table[x_axis], team_table[y_axis], zorder=3, s=0, 
           fc='#1565C0', ec="#000000", alpha=0.70, lw=0.5)

# Adding team logos on scatter points
fotmob_url = 'https://images.fotmob.com/image_resources/logo/teamlogo/'
logo_size = 0.03  # Size of the logo relative to the plot

# Check if fotmob_id column exists in the dataframe
if 'fotmob_id' in team_table.columns:
    for idx, row in team_table.iterrows():
        try:
            team_id = row['fotmob_id']
            x_pos = row[x_axis]
            y_pos = row[y_axis]
            
            # Load the team logo
            team_logo = Image.open(urllib.request.urlopen(f"{fotmob_url}{team_id}.png")).convert('RGBA')
            
            # Create offset box for the logo
            imagebox = OffsetImage(team_logo, zoom=0.15)  # Adjust zoom for logo size
            ab = AnnotationBbox(imagebox, (x_pos, y_pos), frameon=False, zorder=4)
            ax.add_artist(ab)
            
        except Exception as e:
            # If logo loading fails, fall back to text annotation for outliers
            x_threshold = team_table[x_axis].quantile(0.95)
            y_threshold = team_table[y_axis].quantile(0.95)
            if row[x_axis] >= x_threshold or row[y_axis] >= y_threshold:
                ax.annotate(row['team_name'],
                           xy=(x_pos, y_pos),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, color='#000000',
                           bbox=dict(boxstyle='round,pad=0.3', fc='#FFFFFF', ec='#ACA7A5', alpha=0.7, lw=0.5))
else:
    # If fotmob_id column doesn't exist, use text annotations for outliers only
    st.warning("⚠️ Note: 'fotmob_id' column not found in your CSV. Using text labels instead of team logos. Please add a 'fotmob_id' column to your data to display team logos.")
    x_threshold = team_table[x_axis].quantile(0.95)
    y_threshold = team_table[y_axis].quantile(0.95)
    outliers = team_table[(team_table[x_axis] >= x_threshold) | 
                          (team_table[y_axis] >= y_threshold)]
    
    for idx, row in outliers.iterrows():
        ax.annotate(row['team_name'],
                    xy=(row[x_axis], row[y_axis]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, color='#000000',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#FFFFFF', ec='#ACA7A5', alpha=0.7, lw=0.5))

# Format axis labels (replace underscores with spaces and capitalize)
x_label = x_axis.replace('_', ' ').title()
y_label = y_axis.replace('_', ' ').title()

# Set axis labels
ax.set_xlabel(x_label, fontsize=12, color='#4E616C', fontfamily=fe_regular.name)
ax.set_ylabel(y_label, fontsize=12, color='#4E616C', fontfamily=fe_regular.name)

# Add title based on selected metrics
competitions_str = ', '.join(selected_competitions)
ax.set_title(f'{y_label} vs {x_label}\n{competitions_str}', fontsize=14, color='#000000', pad=15, 
             fontfamily=fe_semibold.name, weight='semibold')

# Tick parameters
ax.tick_params(axis='both', labelsize=10, color='#ACA7A5', labelcolor='#ACA7A5')

plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)

# ===== FIRST CONTACT RATIO ON THROW-INS INTO BOX =====
st.write("---")  # Divider
st.write("### First Contact Won on Throw-Ins Into Box")
st.write("Top 20 teams by volume of throw-ins into box per game")

# Check if there's data
if len(first_contact_data) == 0:
    st.warning("No throw-in data available for first contact analysis.")
else:
    # Sort by volume and get top 20
    first_contact_by_volume = first_contact_data.sort_values(by='set_pieces_per_game', ascending=False).head(20)
    # Sort by first contact ratio (removed redundant .head(20))
    plot_data_fc = first_contact_by_volume.sort_values(by='first_contact_ratio', ascending=False)
    
    # Create figure and axis
    fig_fc, ax_fc = plt.subplots(figsize=(12, 10))
    fig_fc.patch.set_facecolor('#D7D1CF')
    ax_fc.set_facecolor('#D7D1CF')
    
    # Create horizontal bars
    y_pos = np.arange(len(plot_data_fc))
    bars = ax_fc.barh(y_pos, plot_data_fc['first_contact_ratio'], color='#D32F2F', 
                      edgecolor='black', linewidth=0.8, zorder=2)
    
    # Set y-axis labels (team names)
    ax_fc.set_yticks(y_pos)
    ax_fc.set_yticklabels(plot_data_fc['team_name'], fontsize=11, fontfamily=fe_regular.name)
    ax_fc.invert_yaxis()  # Invert to show top performers at the top
    ax_fc.grid(lw=1, color="#ACA7A5", axis='x', ls="--")
    
    # Set x-axis
    ax_fc.set_xlim(0, plot_data_fc['first_contact_ratio'].max() * 1.15)
    ax_fc.set_xlabel('')
    
    # Remove top and right spines
    ax_fc.spines['top'].set_visible(False)
    ax_fc.spines['right'].set_visible(False)
    ax_fc.spines['left'].set_visible(False)
    
    # Add set_pieces_per_game values on the right side
    for i, (idx, row) in enumerate(plot_data_fc.iterrows()):
        # Add circle with value
        circle_x = plot_data_fc['first_contact_ratio'].max() * 1.1
        ax_fc.text(circle_x, i, f"{row['set_pieces_per_game']:.1f}", 
                ha='center', va='center', fontsize=11, fontweight='bold',
                fontfamily=fe_regular.name,
                bbox=dict(boxstyle='circle,pad=0.3', facecolor='#D7D1CF', 
                         edgecolor='black', linewidth=1.5))
    
    # Add title and subtitle
    competitions_str = ', '.join(selected_competitions)
    formatted_season_display = set_pieces['formatted_season'].iloc[0] if 'formatted_season' in set_pieces.columns and len(set_pieces) > 0 else ''
    season_text = f" {formatted_season_display}" if formatted_season_display else ""
    
    ax_fc.text(0, len(plot_data_fc) - 22.3, 'Ratio of first contacts won on throw ins into box', 
              fontsize=20, fontweight='bold', ha='left', fontfamily=fe_semibold.name,
              transform=ax_fc.transData)
    ax_fc.text(0, len(plot_data_fc) - 21.5, 
              f'Top 20 teams for volume of throw ins into box per game\n{competitions_str}{season_text}',
              fontsize=11, color='#4E616C', ha='left', fontfamily=fe_regular.name,
              transform=ax_fc.transData)
    
    # Add label on the right
    ax_fc.text(plot_data_fc['first_contact_ratio'].max() * 1.1, len(plot_data_fc) - 21.5, 
              'Throw ins into the box\nper game', ha='center', fontsize=9, color='#4E616C',
              fontfamily=fe_regular.name, transform=ax_fc.transData)
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig_fc)

# ===== VAEP DIFFERENCE BOX PLOT =====
st.write("---")  # Divider
st.write("### Throw-In Value Creation in Final Third")
st.write("Distribution showing danger creation or possession duration after throw-ins")

# Add selector for analysis type
throw_in_filter = st.radio(
    "Select analysis type:",
    options=[
        "All final third throw-ins (VAEP)", 
        "Final third throw-ins excluding box (VAEP)", 
        "Possession duration after final third throw-ins"
    ],
    index=0,
    horizontal=False
)

# Filter data based on selection
if throw_in_filter == "Possession duration after final third throw-ins":
    # Use set_pieces for possession duration
    df1 = set_pieces[set_pieces['start_x_a0'] >= 75]
    
    # Check if possession_chain_duration column exists
    if 'possession_chain_duration' not in df1.columns:
        st.error("'possession_chain_duration' column not found in the dataset.")
        df1 = pd.DataFrame()  # Empty dataframe
    else:
        title_text = 'How long do teams keep possession after a final third throw in?'
        subtitle_text = 'Distribution of length of possession in seconds after throw in.'
        y_label = 'Possession Duration (seconds)'
        metric_column = 'possession_chain_duration'
else:
    # Use atomic data for VAEP analysis
    df1a = dfa_atomic[dfa_atomic['type_name'] == 'throw_in']
    df1b = df1a[df1a['x_a0'] >= 75].dropna()
    
    if throw_in_filter == "All final third throw-ins (VAEP)":
        df1 = df1b
        title_text = 'How much danger do teams create in the 5 seconds after a throw in in the final third?'
    else:  # "Final third throw-ins excluding box (VAEP)"
        df1 = df1b[df1b['is_inbox'] != True]
        title_text = 'How much danger do teams create in the 5 seconds after a throw in in the final third (excluding those into the box)?'
    
    subtitle_text = 'Distribution of difference in Atomic VAEP value between throw in and the highest value in the following actions in a 5 seconds window.'
    y_label = 'VAEP Difference'
    metric_column = 'vaep_difference'

# Check if there's data
if len(df1) == 0:
    st.warning("No throw-in data available for the selected analysis.")
else:
    # Calculate median for ranking
    team_medians = df1.groupby('team_name')[metric_column].median().sort_values(ascending=False)
    
    # For possession duration, show only top 20 teams
    if throw_in_filter == "Possession duration after final third throw-ins":
        teams = team_medians.nlargest(20).index.tolist()
    else:
        teams = team_medians.index.tolist()
    
    # Prepare data for box plot
    data_to_plot = []
    labels = []
    counts = []
    
    for team in teams:
        team_data = df1[df1['team_name'] == team][metric_column].dropna()
        if len(team_data) > 0:  # Only add if there's data
            data_to_plot.append(team_data.values)  # Convert to numpy array
            labels.append(team)
            counts.append(len(team_data))
    
    if len(data_to_plot) == 0:
        st.warning("No valid data to display.")
    else:
        # Create figure
        fig_vaep, ax_vaep = plt.subplots(figsize=(16, 8))
        fig_vaep.patch.set_facecolor('#D7D1CF')
        ax_vaep.set_facecolor('#D7D1CF')
        
        # Create box plots
        bp = ax_vaep.boxplot(data_to_plot, 
                        positions=range(len(data_to_plot)),
                        widths=0.6,
                        patch_artist=True,
                        showfliers=False,
                        showcaps=False,
                        whiskerprops=dict(visible=False),
                        manage_ticks=True)
        
        # Create gradient colors from red to blue
        red_color = np.array(mcolors.to_rgb('#D32F2F'))
        blue_color = np.array(mcolors.to_rgb('#1565C0'))
        colors = []
        n_boxes = len(bp['boxes'])
        
        for i in range(n_boxes):
            # Linear interpolation from red (i=0) to blue (i=n_boxes-1)
            ratio = i / (n_boxes - 1) if n_boxes > 1 else 0
            color = (1 - ratio) * red_color + ratio * blue_color
            colors.append(color)
        
        # Apply gradient colors to boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        # Style the median lines
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        
        # Calculate appropriate y-axis limits based on quartiles
        all_q1 = [np.percentile(data, 25) for data in data_to_plot]
        all_q3 = [np.percentile(data, 75) for data in data_to_plot]
        y_min = min(all_q1)
        y_max = max(all_q3)
        y_range = y_max - y_min
        padding = y_range * 0.15  # 15% padding
        ax_vaep.set_ylim(y_min - padding, y_max + padding)
        
        # Set labels with counts below
        ax_vaep.set_xticks(range(len(labels)))
        ax_vaep.set_xticklabels([f'{label}\n({count})' for label, count in zip(labels, counts)], 
                            rotation=45, ha='right', fontsize=10, fontfamily=fe_regular.name)
        ax_vaep.set_ylabel(y_label, fontsize=12, fontweight='bold', fontfamily=fe_regular.name)
        
        # Add both titles
        competitions_str = ', '.join(selected_competitions)
        formatted_season_display = set_pieces['formatted_season'].iloc[0] if 'formatted_season' in set_pieces.columns and len(set_pieces) > 0 else ''
        season_text = f" {formatted_season_display}" if formatted_season_display else ""
        
        # Calculate title x position based on title length (FIXED: now actually uses the calculated value)
        title_x = 0.0355 if throw_in_filter == "Possession duration after final third throw-ins" else 0.046
        
        fig_vaep.suptitle(title_text, 
                 fontsize=20, x=title_x, ha='left', fontfamily=fe_semibold.name)
        ax_vaep.set_title(f'{subtitle_text}\n' +
                 f'Upper and lower limits of box represent upper and lower quartiles respectively. | Outliers removed from visualisation. {competitions_str}{season_text}', 
                 fontsize=12, loc='left', pad=20, fontfamily=fe_regular.name)
        
        # Add reference line at 0
        ax_vaep.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add grid
        ax_vaep.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax_vaep.set_axisbelow(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Display the plot in Streamlit
        st.pyplot(fig_vaep)

# ===== TEAM-SPECIFIC THROW-IN VISUALIZATION =====
st.write("---")  # Divider
st.write("### Team-Specific Throw-In Analysis")
st.write("Select a team to visualize their final third throw-in patterns")

# Team selector
available_teams = sorted(team_table['team_name'].unique().tolist())

if len(available_teams) == 0:
    st.warning("No teams available for the selected competition(s).")
else:
    selected_team = st.selectbox(
        'Select Team:',
        options=available_teams,
        index=0
    )

    if selected_team:
        # Filter throw-ins for selected team
        setp0 = set_pieces[set_pieces["team_name"] == selected_team]
        setp1 = setp0[setp0["type_name"].isin(['throw_in'])]
        setp2 = setp1[setp1["start_x_a0"] >= 75]  # Final third only
        setpa = setp2[setp2["start_y_a0"] >= 34]  # Upper half
        setpb = setp2[setp2["start_y_a0"] < 34]   # Lower half
        
        # Check if team has throw-ins in final third
        if len(setp2) == 0:
            st.warning(f"{selected_team} has no throw-ins in the final third for the selected competition(s).")
        else:
            # Set up figure
            fig2 = plt.figure(figsize=(20, 12), constrained_layout=True, facecolor='#D7D1CF')
            gs = fig2.add_gridspec(3, 6, wspace=0.1, hspace=0.1)
            
            # Create the axes
            ax1 = fig2.add_subplot(gs[:, :3]) 
            ax2 = fig2.add_subplot(gs[0:3, 3:])
            
            # Create the pitches
            pitch1 = VerticalPitch(pitch_type='custom', pitch_width=68, pitch_length=105, half=True, 
                                   pad_top=0.4, goal_type='box', pad_bottom=0.4,
                                   linewidth=1.25, line_color='#000000', pitch_color='#D7D1CF')
            pitch2 = VerticalPitch(pitch_type='custom', pitch_width=68, pitch_length=105, half=True, 
                                   pad_top=0.4, goal_type='box', pad_bottom=0.4,
                                   linewidth=1.25, line_color='#000000', pitch_color='#D7D1CF')
            
            # Set background color for axes
            ax1.set_facecolor('#D7D1CF')
            ax2.set_facecolor('#D7D1CF')
            
            pitch1.draw(ax=ax1)
            pitch2.draw(ax=ax2)
            
            # Add dotted lines at x = 75 on both pitches
            ax1.axhline(y=75, color='#000000', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
            ax2.axhline(y=75, color='#000000', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
            
            # Create heatmaps
            bins = (18, 12)
            cmapx = mcolors.LinearSegmentedColormap.from_list("custom_red", ["#D7D1CF", "#1565C0"])
            
            if len(setpa) > 0:
                bs_heatmap = pitch1.bin_statistic(setpa.end_x_a0, setpa.end_y_a0, statistic='count', bins=bins)
                hm = pitch1.heatmap(bs_heatmap, ax=ax1, cmap=cmapx, zorder=0, alpha=0.6)
                
                # Arrows and scatter for upper half
                pitch1.arrows(setpa.start_x_a0, setpa.start_y_a0, setpa.end_x_a0, setpa.end_y_a0, 
                             width=1.5, zorder=1, alpha=0.5,
                             ec='#FFFFFF', fc='#000000', headwidth=10, headlength=8, ax=ax1)
                pitch1.scatter(setpa.start_x_a0, setpa.start_y_a0, c='#000000', marker='o', s=100, ax=ax1, zorder=1)
            
            if len(setpb) > 0:
                bs_heatmapy = pitch2.bin_statistic(setpb.end_x_a0, setpb.end_y_a0, statistic='count', bins=bins)
                hm = pitch2.heatmap(bs_heatmapy, ax=ax2, cmap=cmapx, zorder=0, alpha=0.6)
                
                # Arrows and scatter for lower half
                pitch2.arrows(setpb.start_x_a0, setpb.start_y_a0, setpb.end_x_a0, setpb.end_y_a0, 
                             width=1.5, zorder=1, alpha=0.5,
                             ec='#FFFFFF', fc='#000000', headwidth=10, headlength=8, ax=ax2)
                pitch2.scatter(setpb.start_x_a0, setpb.start_y_a0, c='#000000', marker='o', s=100, ax=ax2, zorder=1)
            
            # Titles for both subplots
            competition_ids = ', '.join(setp1['competition_id'].unique())
            formatted_season = setp1['formatted_season'].iloc[0] if 'formatted_season' in setp1.columns and len(setp1) > 0 else ''
            season_display = f" {formatted_season}" if formatted_season else ""
            
            ax1.text(0.5, 1.05, f"Number of throw ins: {setpa.shape[0]}",
                     color='#000000', va='center', ha='center', fontsize=11, transform=ax1.transAxes,
                     fontfamily=fe_regular.name)
            ax2.text(0.5, 1.05, f"Number of throw ins: {setpb.shape[0]}",
                     color='#000000', va='center', ha='center', fontsize=11, transform=ax2.transAxes,
                     fontfamily=fe_regular.name)
            
            # Main title
            fig2.text(0.07, 0.905, f'{selected_team} final third throw ins map', 
                     fontsize=30, va='center', ha='left', fontfamily=fe_semibold.name)
            fig2.text(0.07, 0.87, f"Heatmap: Ending Coordinates | {competition_ids}{season_display}", 
                     fontsize=20, va='center', ha='left', fontfamily=fe_regular.name)
            fig2.text(0, 0.15, 'X: @gualanodavide | Bluesky: @gualanodavide.bsky.social | Linkedin: www.linkedin.com/in/davide-gualano-a2454b187 | Newsletter: the-cutback.beehiiv.com', 
                     va='center', ha='left', fontsize=12, fontfamily=fe_regular.name)
            
            # Adding the club logo
            if 'fotmob_id' in team_table.columns:
                try:
                    team_id = team_table[team_table['team_name'] == selected_team]['fotmob_id'].iloc[0]
                    logo_x = 0.01
                    logo_y = 0.86
                    logo_size = 0.06
                    
                    image_ax = fig2.add_axes([logo_x, logo_y, logo_size, logo_size], fc='None', anchor='C')
                    fotmob_url = 'https://images.fotmob.com/image_resources/logo/teamlogo/'
                    player_face = Image.open(urllib.request.urlopen(f"{fotmob_url}{team_id}.png")).convert('RGBA')
                    image_ax.imshow(player_face)
                    image_ax.axis("off")
                except Exception as e:
                    st.warning(f"Could not load logo for {selected_team}")
            
            # Display the plot in Streamlit
            st.pyplot(fig2)

# ===== PLAYER THROW-IN STATISTICS =====
st.write("---")  # Divider
st.write("### Player Throw-In Statistics")
st.write("Top 15 players by upper quartile throwing length (minimum 10 throw-ins)")

# Filter set_pieces for player statistics (already filtered by competition above)
player_data = set_pieces.copy()

# Calculate player statistics
player_set_piecesa = player_data.groupby('player_name').size().reset_index(name='total_throw_ins')
player_set_piecesb = player_data.groupby('player_name')['length'].quantile(0.75).reset_index(name='upper_quartile_throwing_length')
player_set_piecesc = player_data.groupby('player_name')['length'].max().reset_index(name='maximum_throwing_length')
player_set_piecesd = player_data.groupby('player_name')['length'].mean().reset_index(name='average_throwing_length')
player_set_pieces0 = player_set_piecesa.merge(player_set_piecesb, on='player_name', how='left').fillna(0)
player_set_pieces1 = player_set_pieces0.merge(player_set_piecesc, on='player_name', how='left').fillna(0)
player_set_pieces = player_set_pieces1.merge(player_set_piecesd, on='player_name', how='left').fillna(0)

# Filter for players with at least 10 throw-ins
player_set_pieces = player_set_pieces[player_set_pieces['total_throw_ins'] >= 10]

# Check if there are any players
if len(player_set_pieces) == 0:
    st.warning("No players have 10 or more throw-ins in the selected competition(s).")
else:
    # Sort and get top 15
    plot_data = player_set_pieces.sort_values(by='upper_quartile_throwing_length', ascending=False).head(15)
    
    # Create figure and axis
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    fig3.patch.set_facecolor('#D7D1CF')
    ax3.set_facecolor('#D7D1CF')
    
    # Create horizontal bars
    y_pos = np.arange(len(plot_data))
    bars_max = ax3.barh(y_pos, plot_data['maximum_throwing_length'], color=[0,0,0,0], 
                        edgecolor='black', linewidth=0.8, zorder=2)
    bars_quartile = ax3.barh(y_pos, plot_data['upper_quartile_throwing_length'], 
                             color='#D32F2F', edgecolor='black', linewidth=0.8, zorder=2)
    
    # Set y-axis labels (player names)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(plot_data['player_name'], fontsize=11, fontfamily=fe_regular.name)
    ax3.invert_yaxis()  # Invert to show top performers at the top
    ax3.grid(lw=1, color="#ACA7A5", axis='x', ls="--")
    
    # Set x-axis
    ax3.set_xlim(0, plot_data['maximum_throwing_length'].max() * 1.15)
    ax3.set_xlabel('')
    
    # Remove top and right spines
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    
    # Add average_throwing_length values on the right side
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        # Add circle with value
        circle_x = plot_data['maximum_throwing_length'].max() * 1.1
        ax3.text(circle_x, i, f"{row['average_throwing_length']:.2f}", 
                ha='center', va='center', fontsize=9, fontweight='bold',
                fontfamily=fe_regular.name,
                bbox=dict(boxstyle='circle,pad=0.3', facecolor='#D7D1CF', 
                         edgecolor='black', linewidth=1.5))
    
    # Add title and subtitle
    competitions_str = ', '.join(selected_competitions)
    formatted_season_display = set_pieces['formatted_season'].iloc[0] if 'formatted_season' in set_pieces.columns and len(set_pieces) > 0 else ''
    season_text = f" {formatted_season_display}" if formatted_season_display else ""
    
    ax3.text(0, len(plot_data) - 17, 'Quarterbacks wannabe', 
            fontsize=20, fontweight='bold', ha='left', fontfamily=fe_semibold.name,
            transform=ax3.transData)
    ax3.text(0, len(plot_data) - 16.2, 
            f'Top 15 players for upper quartile of throw in length | minimum 10 throw ins\n{competitions_str}{season_text}',
            fontsize=11, color='#4E616C', ha='left', fontfamily=fe_regular.name,
            transform=ax3.transData)
    
    # Add "Avg. throwing length" label on the right
    ax3.text(plot_data['maximum_throwing_length'].max() * 1.1, len(plot_data) - 16.2, 
            'Avg. throwing length (m)', ha='center', fontsize=9, color='#4E616C',
            fontfamily=fe_regular.name, transform=ax3.transData)
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig3)
