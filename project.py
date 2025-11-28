from flask import Flask, render_template, request, jsonify
import fastf1
from fastf1.ergast import Ergast
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import os
from datetime import datetime

app = Flask(__name__)

# Enable FastF1 cache
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('./cache')

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return plot_url

def get_seasons():
    """Get available seasons"""
    return list(range(2018, datetime.now().year + 1))

def get_events(season):
    """Get events for a season"""
    try:
        schedule = fastf1.events.get_event_schedule(season, backend='ergast')
        return [(row['RoundNumber'], f"R{row['RoundNumber']}: {row['EventName']}") 
                for _, row in schedule.iterrows()]
    except:
        return []

def get_session_types():
    """Get available session types"""
    return [('FP1', 'Free Practice 1'), ('FP2', 'Free Practice 2'), 
            ('FP3', 'Free Practice 3'), ('Q', 'Qualifying'), ('R', 'Race'), ('S', 'Sprint')]

def load_session(year, round_num, session_type):
    """Load a session safely"""
    try:
        session = fastf1.get_session(year, round_num, session_type)
        session.load()
        return session
    except Exception as e:
        print(f"Error loading session: {e}")
        return None

def get_theme_colors(theme):
    """Get colors for a specific theme"""
    themes = {
        'default': {'primary': '#FF6900', 'secondary': '#FFD700', 'accent': '#00D9FF', 
                   'plot_bg': '#1a1a1a', 'plot_fg': '#2a2a2a'},
        'red_bull': {'primary': '#0600EF', 'secondary': '#DC0000', 'accent': '#FCD700',
                    'plot_bg': '#1a1a2e', 'plot_fg': '#2a2a3e'},
        'ferrari': {'primary': '#DC0000', 'secondary': '#FFF500', 'accent': '#FFFFFF',
                   'plot_bg': '#2e1a1a', 'plot_fg': '#3e2a2a'},
        'mercedes': {'primary': '#00D2BE', 'secondary': '#C0C0C0', 'accent': '#000000',
                    'plot_bg': '#1a2e2c', 'plot_fg': '#2a3e3c'},
        'mclaren': {'primary': '#FF8700', 'secondary': '#47C7FC', 'accent': '#FFFFFF',
                   'plot_bg': '#2e2a1a', 'plot_fg': '#3e3a2a'},
        'aston_martin': {'primary': '#006F62', 'secondary': '#CEDC00', 'accent': '#FFFFFF',
                        'plot_bg': '#1a2e2a', 'plot_fg': '#2a3e3a'},
        'alpine': {'primary': '#0090FF', 'secondary': '#FF1E7A', 'accent': '#FFFFFF',
                  'plot_bg': '#1a2a3e', 'plot_fg': '#2a3a4e'},
        'williams': {'primary': '#005AFF', 'secondary': '#00A0DE', 'accent': '#FFFFFF',
                    'plot_bg': '#1a1a3e', 'plot_fg': '#2a2a4e'},
        'rb': {'primary': '#2B4562', 'secondary': '#6692FF', 'accent': '#FFFFFF',
              'plot_bg': '#1a2530', 'plot_fg': '#2a3540'},
        'kick_sauber': {'primary': '#00E701', 'secondary': '#52E500', 'accent': '#FFFFFF',
                       'plot_bg': '#1a2e1a', 'plot_fg': '#2a3e2a'},
        'haas': {'primary': '#B6BABD', 'secondary': '#ED1C24', 'accent': '#FFFFFF',
                'plot_bg': '#2a2a2a', 'plot_fg': '#3a3a3a'}
    }
    return themes.get(theme, themes['default'])


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/seasons')
def api_seasons():
    return jsonify(get_seasons())

@app.route('/api/events/<int:season>')
def api_events(season):
    events = get_events(season)
    return jsonify(events)

@app.route('/api/drivers/<int:year>/<int:round_num>/<session_type>')
def api_drivers(year, round_num, session_type):
    session = load_session(year, round_num, session_type)
    if session is None:
        return jsonify([])
    
    drivers = []
    for driver in session.drivers:
        try:
            # Get driver abbreviation from the laps data
            driver_laps = session.laps.pick_driver(driver)
            if len(driver_laps) > 0:
                driver_abbr = driver_laps.iloc[0]['Driver']
                drivers.append({
                    'code': driver_abbr,
                    'number': driver,
                    'name': driver_abbr
                })
        except Exception as e:
            print(f"Error processing driver {driver}: {e}")
            pass
    
    return jsonify(drivers)

@app.route('/api/teams/<int:year>/<int:round_num>/<session_type>')
def api_teams(year, round_num, session_type):
    session = load_session(year, round_num, session_type)
    if session is None:
        return jsonify([])
    
    teams = session.laps['Team'].unique().tolist()
    return jsonify(sorted([t for t in teams if pd.notna(t)]))

@app.route('/plot/speed-comparison', methods=['POST'])
def plot_speed_comparison():
    data = request.json
    year = data.get('year')
    round_num = data.get('round')
    session_type = data.get('session_type')
    drivers = data.get('drivers', [])
    theme = data.get('theme', 'default')
    
    if not drivers or len(drivers) < 2:
        return jsonify({'error': 'Select at least 2 drivers'}), 400
    
    try:
        session = load_session(year, round_num, session_type)
        if session is None:
            return jsonify({'error': 'Session not available'}), 400
        
        # Get theme colors
        theme_colors = get_theme_colors(theme)
        
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=theme_colors['plot_bg'])
        ax.set_facecolor(theme_colors['plot_fg'])
        
        # Use more vibrant colors for the lines
        colors = [theme_colors['primary'], theme_colors['secondary'], theme_colors['accent'], 
                  '#FF1493', '#00FF00', '#FFD700', '#8B00FF']
        
        for idx, driver in enumerate(drivers):
            try:
                lap = session.laps.pick_driver(driver).pick_fastest()
                if lap is None or len(lap) == 0:
                    continue
                    
                tel = lap.get_car_data().add_distance()
                ax.plot(tel['Distance'], tel['Speed'], label=driver, 
                       color=colors[idx % len(colors)], linewidth=3, alpha=0.9,
                       marker='o', markersize=0, markevery=50)
            except:
                continue
        
        ax.set_xlabel('Distance (m)', fontsize=13, color='white', fontweight='bold')
        ax.set_ylabel('Speed (km/h)', fontsize=13, color='white', fontweight='bold')
        ax.set_title(f'Speed Comparison - {session.event["Location"]}', 
                    fontsize=16, color='white', pad=25, fontweight='bold')
        
        # Prettier legend
        legend = ax.legend(loc='best', framealpha=0.95, facecolor=theme_colors['plot_fg'],
                          edgecolor=theme_colors['primary'], fontsize=11, 
                          shadow=True, fancybox=True)
        plt.setp(legend.get_texts(), color='white')
        
        # Prettier grid
        ax.grid(True, alpha=0.3, color=theme_colors['primary'], linestyle='--', linewidth=0.8)
        ax.tick_params(colors='white', labelsize=10)
        ax.spines['bottom'].set_color(theme_colors['primary'])
        ax.spines['top'].set_color(theme_colors['primary']) 
        ax.spines['right'].set_color(theme_colors['primary'])
        ax.spines['left'].set_color(theme_colors['primary'])
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        
        plt.tight_layout()
        plot_url = fig_to_base64(fig)
        return jsonify({'plot': plot_url})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/plot/lap-comparison', methods=['POST'])
def plot_lap_comparison():
    data = request.json
    year = data.get('year')
    round_num = data.get('round')
    ref_driver = data.get('ref_driver')
    target_driver = data.get('target_driver')
    theme = data.get('theme', 'default')
    
    if not ref_driver or not target_driver:
        return jsonify({'error': 'Select two drivers'}), 400
    
    try:
        session = load_session(year, round_num, 'R')
        if session is None:
            return jsonify({'error': 'Race session not available'}), 400
        
        laps = session.laps
        ref_laps = laps.pick_driver(ref_driver).dropna(subset=['LapTime'])
        target_laps = laps.pick_driver(target_driver).dropna(subset=['LapTime'])
        
        ref_df = ref_laps.set_index('LapNumber')[['LapTime']]
        ref_df['RefTime'] = ref_df['LapTime'].dt.total_seconds()
        ref_df = ref_df.drop(columns=['LapTime'])
        
        target_df = target_laps.set_index('LapNumber')[['LapTime']]
        target_df['TargetTime'] = target_df['LapTime'].dt.total_seconds()
        target_df = target_df.drop(columns=['LapTime'])
        
        common_laps = ref_df.join(target_df, how='inner')
        common_laps['RefCumulative'] = common_laps['RefTime'].cumsum()
        common_laps['TargetCumulative'] = common_laps['TargetTime'].cumsum()
        common_laps['Gap'] = common_laps['TargetCumulative'] - common_laps['RefCumulative']
        
        # Get theme colors
        theme_colors = get_theme_colors(theme)
        
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=theme_colors['plot_bg'])
        ax.set_facecolor(theme_colors['plot_fg'])
        
        ax.plot(common_laps.index, common_laps['Gap'], label=f'{target_driver} vs {ref_driver}',
               color=theme_colors['accent'], linewidth=3.5, alpha=0.9, zorder=3)
        ax.axhline(0, color='white', linestyle='--', alpha=0.7, linewidth=2)
        ax.fill_between(common_laps.index, common_laps['Gap'], 0, 
                        where=(common_laps['Gap']>=0), alpha=0.4, color='#FF1493', 
                        label='Behind', edgecolor='#FF1493', linewidth=1.5)
        ax.fill_between(common_laps.index, common_laps['Gap'], 0,
                        where=(common_laps['Gap']<0), alpha=0.4, color='#00FF00', 
                        label='Ahead', edgecolor='#00FF00', linewidth=1.5)
        
        ax.set_xlabel('Lap Number', fontsize=13, color='white', fontweight='bold')
        ax.set_ylabel('Gap (seconds)', fontsize=13, color='white', fontweight='bold')
        ax.set_title(f'Cumulative Gap - {session.event["Location"]}', 
                    fontsize=16, color='white', pad=25, fontweight='bold')
        
        # Prettier legend
        legend = ax.legend(loc='best', framealpha=0.95, facecolor=theme_colors['plot_fg'],
                          edgecolor=theme_colors['primary'], fontsize=11, 
                          shadow=True, fancybox=True)
        plt.setp(legend.get_texts(), color='white')
        
        # Prettier grid
        ax.grid(True, alpha=0.3, color=theme_colors['primary'], linestyle='--', linewidth=0.8)
        ax.tick_params(colors='white', labelsize=10)
        ax.spines['bottom'].set_color(theme_colors['primary'])
        ax.spines['top'].set_color(theme_colors['primary']) 
        ax.spines['right'].set_color(theme_colors['primary'])
        ax.spines['left'].set_color(theme_colors['primary'])
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        
        plt.tight_layout()
        plot_url = fig_to_base64(fig)
        return jsonify({'plot': plot_url})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/plot/tyre-strategy', methods=['POST'])
def plot_tyre_strategy():
    data = request.json
    year = data.get('year')
    round_num = data.get('round')
    theme = data.get('theme', 'default')
    
    try:
        session = load_session(year, round_num, 'R')
        if session is None:
            return jsonify({'error': 'Race session not available'}), 400
        
        laps = session.laps
        stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
        stints = stints.groupby(["Driver", "Stint", "Compound"]).count().reset_index()
        stints = stints.rename(columns={"LapNumber": "StintLength"})
        
        drivers = session.drivers
        drivers = [session.laps.pick_driver(d).iloc[0]['Driver'] for d in drivers if len(session.laps.pick_driver(d)) > 0]
        
        # Get theme colors
        theme_colors = get_theme_colors(theme)
        
        fig, ax = plt.subplots(figsize=(14, 11), facecolor=theme_colors['plot_bg'])
        ax.set_facecolor(theme_colors['plot_fg'])
        
        compound_colors = {
            'SOFT': '#FF1493',
            'MEDIUM': '#FFD700',
            'HARD': '#FFFFFF',
            'INTERMEDIATE': '#00FF00',
            'WET': '#0099FF'
        }
        
        for driver in drivers:
            driver_stints = stints.loc[stints["Driver"] == driver]
            previous_stint_end = 0
            
            for _, row in driver_stints.iterrows():
                compound_color = compound_colors.get(row["Compound"], '#888888')
                ax.barh(y=driver, width=row["StintLength"], left=previous_stint_end,
                       color=compound_color, edgecolor=theme_colors['primary'], 
                       fill=True, linewidth=2, alpha=0.9)
                previous_stint_end += row["StintLength"]
        
        ax.set_xlabel('Lap Number', fontsize=13, color='white', fontweight='bold')
        ax.set_title(f'{session.event["Location"]} - Tyre Strategies', 
                    fontsize=16, color='white', pad=25, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, color=theme_colors['primary'], linestyle='--', linewidth=0.8)
        ax.tick_params(colors='white', labelsize=10)
        ax.spines['bottom'].set_color(theme_colors['primary'])
        ax.spines['top'].set_color(theme_colors['primary']) 
        ax.spines['right'].set_color(theme_colors['primary'])
        ax.spines['left'].set_color(theme_colors['primary'])
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        
        # Add legend for compounds
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, edgecolor=theme_colors['primary'], 
                                label=compound, linewidth=2) 
                          for compound, color in compound_colors.items()]
        legend = ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95, 
                          facecolor=theme_colors['plot_fg'], edgecolor=theme_colors['primary'],
                          fontsize=11, shadow=True, fancybox=True)
        plt.setp(legend.get_texts(), color='white')
        
        plt.tight_layout()
        plot_url = fig_to_base64(fig)
        return jsonify({'plot': plot_url})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/plot/team-pace', methods=['POST'])
def plot_team_pace():
    data = request.json
    year = data.get('year')
    round_num = data.get('round')
    theme = data.get('theme', 'default')
    
    try:
        session = load_session(year, round_num, 'R')
        if session is None:
            return jsonify({'error': 'Race session not available'}), 400
        
        laps = session.laps.pick_quicklaps()
        transformed_laps = laps.copy()
        transformed_laps['LapTime (s)'] = laps['LapTime'].dt.total_seconds()
        
        team_order = (
            transformed_laps[["Team", "LapTime (s)"]].groupby("Team")
            .median()["LapTime (s)"].sort_values().index
        )
        
        # Get theme colors
        theme_colors = get_theme_colors(theme)
        
        fig, ax = plt.subplots(figsize=(15, 9), facecolor=theme_colors['plot_bg'])
        ax.set_facecolor(theme_colors['plot_fg'])
        
        team_data = [transformed_laps[transformed_laps['Team'] == team]['LapTime (s)'].values 
                     for team in team_order]
        
        bp = ax.boxplot(team_data, labels=team_order, patch_artist=True, vert=True,
                       widths=0.6, showfliers=True,
                       boxprops=dict(linewidth=2, edgecolor=theme_colors['primary']),
                       whiskerprops=dict(linewidth=2, color=theme_colors['primary']),
                       capprops=dict(linewidth=2, color=theme_colors['primary']),
                       medianprops=dict(linewidth=3, color=theme_colors['accent']),
                       flierprops=dict(marker='o', markerfacecolor=theme_colors['secondary'], 
                                      markersize=6, alpha=0.6, markeredgecolor=theme_colors['primary']))
        
        for patch, team in zip(bp['boxes'], team_order):
            try:
                team_color = fastf1.plotting.get_team_color(team, session=session)
                patch.set_facecolor(team_color)
                patch.set_alpha(0.8)
            except:
                patch.set_facecolor('#888888')
                patch.set_alpha(0.8)
        
        ax.set_ylabel('Lap Time (seconds)', fontsize=13, color='white', fontweight='bold')
        ax.set_title(f'{session.event["Location"]} - Team Pace Distribution', 
                    fontsize=16, color='white', pad=25, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, color=theme_colors['primary'], linestyle='--', linewidth=0.8)
        ax.tick_params(colors='white', labelsize=10)
        ax.spines['bottom'].set_color(theme_colors['primary'])
        ax.spines['top'].set_color(theme_colors['primary']) 
        ax.spines['right'].set_color(theme_colors['primary'])
        ax.spines['left'].set_color(theme_colors['primary'])
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)
        
        plt.tight_layout()
        plot_url = fig_to_base64(fig)
        return jsonify({'plot': plot_url})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/plot/laptime-distribution', methods=['POST'])
def plot_laptime_distribution():
    data = request.json
    year = data.get('year')
    round_num = data.get('round')
    drivers = data.get('drivers', [])
    theme = data.get('theme', 'default')
    
    if not drivers or len(drivers) < 1:
        return jsonify({'error': 'Select at least 1 driver'}), 400
    
    try:
        session = load_session(year, round_num, 'R')
        if session is None:
            return jsonify({'error': 'Race session not available'}), 400
        
        # Get theme colors
        theme_colors = get_theme_colors(theme)
        
        # Get laps for selected drivers and filter quick laps
        driver_laps = session.laps.pick_drivers(drivers).pick_quicklaps().reset_index()
        
        if len(driver_laps) == 0:
            return jsonify({'error': 'No lap data available for selected drivers'}), 400
        
        # Convert timedelta to seconds
        driver_laps['LapTime(s)'] = driver_laps['LapTime'].dt.total_seconds()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(10, len(drivers) * 1.5), 8), 
                              facecolor=theme_colors['plot_bg'])
        ax.set_facecolor(theme_colors['plot_fg'])
        
        # Get driver colors
        driver_colors = {}
        for driver in drivers:
            try:
                driver_colors[driver] = fastf1.plotting.get_driver_color(driver, session=session)
            except:
                driver_colors[driver] = theme_colors['primary']
        
        # Create violin plot
        parts = ax.violinplot(
            [driver_laps[driver_laps['Driver'] == driver]['LapTime(s)'].values for driver in drivers],
            positions=range(len(drivers)),
            widths=0.7,
            showmeans=False,
            showmedians=True,
            showextrema=True
        )
        
        # Color the violins
        for idx, pc in enumerate(parts['bodies']):
            driver = drivers[idx]
            pc.set_facecolor(driver_colors[driver])
            pc.set_alpha(0.7)
            pc.set_edgecolor(theme_colors['primary'])
            pc.set_linewidth(2)
        
        # Style the median, extrema lines
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if partname in parts:
                vp = parts[partname]
                vp.set_edgecolor(theme_colors['accent'])
                vp.set_linewidth(2)
        
        # Add scatter plot for individual laps colored by compound
        compound_colors = {
            'SOFT': '#FF1493',
            'MEDIUM': '#FFD700',
            'HARD': '#FFFFFF',
            'INTERMEDIATE': '#00FF00',
            'WET': '#0099FF'
        }
        
        for idx, driver in enumerate(drivers):
            driver_data = driver_laps[driver_laps['Driver'] == driver]
            for compound in driver_data['Compound'].unique():
                if pd.notna(compound):
                    compound_data = driver_data[driver_data['Compound'] == compound]
                    x_positions = np.random.normal(idx, 0.04, size=len(compound_data))
                    ax.scatter(x_positions, compound_data['LapTime(s)'], 
                             c=compound_colors.get(compound, '#888888'),
                             s=30, alpha=0.6, edgecolors=theme_colors['primary'],
                             linewidth=0.5, zorder=3)
        
        # Styling
        ax.set_xticks(range(len(drivers)))
        ax.set_xticklabels(drivers, fontsize=11, color='white', fontweight='bold')
        ax.set_xlabel('Driver', fontsize=13, color='white', fontweight='bold')
        ax.set_ylabel('Lap Time (seconds)', fontsize=13, color='white', fontweight='bold')
        ax.set_title(f'{session.event["Location"]} - Lap Time Distributions', 
                    fontsize=16, color='white', pad=25, fontweight='bold')
        
        # Grid and spines
        ax.grid(axis='y', alpha=0.3, color=theme_colors['primary'], linestyle='--', linewidth=0.8)
        ax.tick_params(colors='white', labelsize=10)
        ax.spines['bottom'].set_color(theme_colors['primary'])
        ax.spines['top'].set_color(theme_colors['primary']) 
        ax.spines['right'].set_color(theme_colors['primary'])
        ax.spines['left'].set_color(theme_colors['primary'])
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        
        # Add legend for compounds
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, edgecolor=theme_colors['primary'], 
                                label=compound, linewidth=1) 
                          for compound, color in compound_colors.items() 
                          if compound in driver_laps['Compound'].values]
        if legend_elements:
            legend = ax.legend(handles=legend_elements, loc='upper right', 
                             framealpha=0.95, facecolor=theme_colors['plot_fg'],
                             edgecolor=theme_colors['primary'], fontsize=10, 
                             shadow=True, fancybox=True, title='Tyre Compound')
            plt.setp(legend.get_texts(), color='white')
            plt.setp(legend.get_title(), color='white', fontweight='bold')
        
        plt.tight_layout()
        plot_url = fig_to_base64(fig)
        return jsonify({'plot': plot_url})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/race-replay-data/<int:year>/<int:round_num>')
def race_replay_data(year, round_num):
    """Get race replay data for visualization"""
    try:
        session = load_session(year, round_num, 'R')
        if session is None:
            return jsonify({'error': 'Race session not available'}), 400
        
        # Get track coordinates from fastest lap
        fastest_lap = session.laps.pick_fastest()
        track_data = fastest_lap.get_pos_data()
        
        # Get all laps
        laps = session.laps
        
        # Build detailed telemetry data
        replay_data = {
            'track': {
                'x': track_data['X'].tolist(),
                'y': track_data['Y'].tolist()
            },
            'drivers': [],
            'event_name': session.event['EventName'],
            'total_laps': int(laps['LapNumber'].max()),
            'frames': []  # Will contain position data for each time frame
        }
        
        # Get data for each driver
        driver_data_map = {}
        for driver_num in session.drivers:
            driver_laps = laps.pick_driver(driver_num)
            if len(driver_laps) == 0:
                continue
                
            driver_abbr = driver_laps.iloc[0]['Driver']
            
            try:
                driver_color = fastf1.plotting.get_driver_color(driver_abbr, session=session)
            except:
                driver_color = '#FFFFFF'
            
            driver_info = {
                'abbr': driver_abbr,
                'number': int(driver_num),
                'color': driver_color,
                'telemetry': []  # Will store position data over time
            }
            
            # Get telemetry for each lap
            for lap_num in range(1, replay_data['total_laps'] + 1):
                lap = driver_laps[driver_laps['LapNumber'] == lap_num]
                if len(lap) > 0:
                    lap = lap.iloc[0]
                    try:
                        # Get position data for this lap
                        tel = lap.get_pos_data()
                        if tel is not None and len(tel) > 0:
                            driver_info['telemetry'].append({
                                'lap': lap_num,
                                'x': tel['X'].tolist(),
                                'y': tel['Y'].tolist(),
                                'distance': tel['Distance'].tolist() if 'Distance' in tel else [],
                                'position': int(lap['Position']) if pd.notna(lap['Position']) else None,
                                'compound': lap['Compound'] if pd.notna(lap['Compound']) else 'UNKNOWN'
                            })
                    except Exception as e:
                        print(f"Error getting telemetry for {driver_abbr} lap {lap_num}: {e}")
                        continue
            
            if len(driver_info['telemetry']) > 0:
                replay_data['drivers'].append(driver_info)
                driver_data_map[driver_abbr] = driver_info
        
        return jsonify(replay_data)
    except Exception as e:
        print(f"Error in race_replay_data: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/championship', methods=['POST'])
def championship():
    data = request.json
    season = data.get('season')
    round_num = data.get('round')
    
    try:
        ergast = Ergast()
        standings = ergast.get_driver_standings(season=season, round=round_num).content[0]
        
        schedule = fastf1.events.get_event_schedule(season, backend='ergast')
        remaining_rounds = schedule[schedule['RoundNumber'] > round_num]
        
        sprint_events = len(remaining_rounds[remaining_rounds['EventFormat'] == 'sprint_qualifying'])
        conventional_events = len(remaining_rounds[remaining_rounds['EventFormat'] == 'conventional'])
        
        POINTS_FOR_SPRINT = 8 + 25
        POINTS_FOR_CONVENTIONAL = 25
        max_points = (sprint_events * POINTS_FOR_SPRINT) + (conventional_events * POINTS_FOR_CONVENTIONAL)
        
        leader_points = int(standings.iloc[0]['points'])
        results = []
        
        for _, driver in standings.iterrows():
            current_pts = int(driver['points'])
            max_pts = current_pts + max_points
            can_win = max_pts >= leader_points
            per_race = (leader_points - current_pts) / (24 - round_num) if round_num < 24 else 0
            
            results.append({
                'position': int(driver['position']),
                'name': f"{driver['givenName']} {driver['familyName']}",
                'points': current_pts,
                'max_points': max_pts,
                'can_win': 'Yes' if can_win else 'No',
                'per_race': round(per_race, 2)
            })
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)