import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

np.set_printoptions(suppress=True)

def visualize_clusters(df, stats_scaled):
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Attack vs Defense', 'Special Attack vs Special Defense', 
                       'Speed vs HP', 'Total Stats Distribution'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'histogram'}]]
    )
    
    df['total_stats'] = df[['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']].sum(axis=1)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    cluster_means = df.groupby('cluster')[['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']].mean()
    cluster_names = {}
    for cluster_id in cluster_means.index:
        cluster_names[cluster_id] = get_cluster_archetype(cluster_means.loc[cluster_id])
    
    #atk vs def
    for cluster in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster]
        hover_text = [
            f"<b>{row['name'].capitalize()}</b><br>" +
            f"<img src='https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{int(row['id'])}.png' width='96'><br>" +
            f"Type: {row['type1']}" + (f"/{row['type2']}" if pd.notna(row['type2']) else "") + "<br>" +
            f"Attack: {row['attack']}<br>" +
            f"Defense: {row['defense']}<br>" +
            f"Total: {int(row['total_stats'])}"
            for _, row in cluster_df.iterrows()
        ]
        
        fig.add_trace(
            go.Scatter(
                x=cluster_df['attack'],
                y=cluster_df['defense'],
                mode='markers',
                name=f"{cluster_names[cluster]}",
                marker=dict(size=10, color=colors[cluster], opacity=0.7),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                legendgroup=f'cluster{cluster}',
                showlegend=True
            ),
            row=1, col=1
        )
    
    #sp atk vs sp def
    for cluster in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster]
        hover_text = [
            f"<b>{row['name'].capitalize()}</b><br>" +
            f"<img src='https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{int(row['id'])}.png' width='96'><br>" +
            f"Type: {row['type1']}" + (f"/{row['type2']}" if pd.notna(row['type2']) else "") + "<br>" +
            f"Sp. Attack: {row['special-attack']}<br>" +
            f"Sp. Defense: {row['special-defense']}<br>" +
            f"Total: {int(row['total_stats'])}"
            for _, row in cluster_df.iterrows()
        ]
        
        fig.add_trace(
            go.Scatter(
                x=cluster_df['special-attack'],
                y=cluster_df['special-defense'],
                mode='markers',
                name=f"{cluster_names[cluster]}",
                marker=dict(size=10, color=colors[cluster], opacity=0.7),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                legendgroup=f'cluster{cluster}',
                showlegend=False
            ),
            row=1, col=2
        )
    
    #spd vs hp
    for cluster in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster]
        hover_text = [
            f"<b>{row['name'].capitalize()}</b><br>" +
            f"<img src='https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{int(row['id'])}.png' width='96'><br>" +
            f"Type: {row['type1']}" + (f"/{row['type2']}" if pd.notna(row['type2']) else "") + "<br>" +
            f"Speed: {row['speed']}<br>" +
            f"HP: {row['hp']}<br>" +
            f"Total: {int(row['total_stats'])}"
            for _, row in cluster_df.iterrows()
        ]
        
        fig.add_trace(
            go.Scatter(
                x=cluster_df['speed'],
                y=cluster_df['hp'],
                mode='markers',
                name=f"{cluster_names[cluster]}",
                marker=dict(size=10, color=colors[cluster], opacity=0.7),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                legendgroup=f'cluster{cluster}',
                showlegend=False
            ),
            row=2, col=1
        )
    
    #total stats
    for cluster in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster]
        fig.add_trace(
            go.Histogram(
                x=cluster_df['total_stats'],
                name=f"{cluster_names[cluster]}",
                marker=dict(color=colors[cluster], opacity=0.7),
                legendgroup=f'cluster{cluster}',
                showlegend=False
            ),
            row=2, col=2
        )
    
    fig.update_xaxes(title_text="Attack", row=1, col=1)
    fig.update_yaxes(title_text="Defense", row=1, col=1)
    
    fig.update_xaxes(title_text="Special Attack", row=1, col=2)
    fig.update_yaxes(title_text="Special Defense", row=1, col=2)
    
    fig.update_xaxes(title_text="Speed", row=2, col=1)
    fig.update_yaxes(title_text="HP", row=2, col=1)
    
    fig.update_xaxes(title_text="Total Base Stats", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    

    fig.update_layout(
        title_text="<b>Pokémon Clusters Visualizer</b>",
        title_x=0.5,
        height=800,
        showlegend=True,
        hovermode='closest',
        template='plotly_white'
    )
    
    fig.write_html('pokemon_clusters.html')
    fig.show()

def get_cluster_archetype(row):
    max_stat = row.idxmax()
    archetypes = {
        'speed': 'Speedy',
        'attack': 'Physical Attacker',
        'special-attack': 'Special Attacker',
        'hp': 'HP Tank',
        'defense': 'Def Tank',
        'special-defense': 'Special Def Tank'
    }

    return archetypes.get(max_stat, 'Balanced')

def recommend_teams(df, num_teams=3):
    
    clusters = df['cluster'].unique()
    
    for team_num in range(1, num_teams + 1):
        print(f"\n{'─'*80}")
        print(f"Team #{team_num}")
        print(f"{'─'*80}")
        
        team = []
        for cluster_id in clusters:
            cluster_pokemon = df[df['cluster'] == cluster_id]
            selected = cluster_pokemon.sample(1).iloc[0]
            team.append(selected)

        team_df = pd.DataFrame(team)
        avg_stats = team_df[['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']].mean()
        
        print(f"\n{'Pokemon':<15} {'Type':<20} {'Role':<20} {'Total Stats':<12}")
        print("─" * 80)
        
        for pokemon in team:
            type_str = pokemon['type1'] if pd.isna(pokemon['type2']) else f"{pokemon['type1']}/{pokemon['type2']}"
            cluster_means = df[df['cluster'] == pokemon['cluster']][['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']].mean()
            role = get_cluster_archetype(cluster_means)
            total = int(pokemon[['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']].sum())
            print(f"{pokemon['name'].capitalize():<15} {type_str:<20} {role:<20} {total:<12}")
        
        print(f"\nTeam Average Stats:")
        print(f"  HP: {avg_stats['hp']:.1f} | ATK: {avg_stats['attack']:.1f} | DEF: {avg_stats['defense']:.1f}")
        print(f"  SP ATK: {avg_stats['special-attack']:.1f} | SP DEF: {avg_stats['special-defense']:.1f} | SPD: {avg_stats['speed']:.1f}")
        print(f"  Total: {avg_stats.sum():.1f}")

if __name__ == '__main__':
    print("🔍 Loading Pokémon data...")
    df = pd.read_csv('Gen1_data.csv')
    
    df = df[df['final_evo'] == True]
    
    stats = df[['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']]

    scaler = StandardScaler()
    scaled_stats = scaler.fit_transform(stats)
    
    print("\nK-means clustering...")
    kmean = KMeans(n_clusters=5, random_state=0)
    kmean.fit(scaled_stats)

    df['cluster'] = kmean.labels_
    
    cluster_means = df.groupby('cluster')[['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']].mean()
    
    for cluster_id in sorted(cluster_means.index):
        row = cluster_means.loc[cluster_id]
        cluster_name = get_cluster_archetype(row)
        cluster_pokemon = df[df['cluster'] == cluster_id]
        
        print(f"\n{'─'*80}")
        print(f"Cluster {cluster_id}: {cluster_name}")
        print(f"{'─'*80}")
        print(f"Number of Pokémon: {len(cluster_pokemon)}")
        print(f"\nAverage Stats:")
        print(f"  HP: {row['hp']:.1f} | Attack: {row['attack']:.1f} | Defense: {row['defense']:.1f}")
        print(f"  Sp.Atk: {row['special-attack']:.1f} | Sp.Def: {row['special-defense']:.1f} | Speed: {row['speed']:.1f}")
        
        print(f"\nPokémon in this cluster:")
        pokemon_list = cluster_pokemon[['name', 'hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']].sort_values('name')
        for idx, pokemon in pokemon_list.iterrows():
            total = int(pokemon[['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']].sum())
            print(f"-{pokemon['name'].capitalize():<15} (Total: {total})")
    
    recommend_teams(df, num_teams=3)
    visualize_clusters(df, scaled_stats)