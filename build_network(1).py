import os
import subprocess
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端，避免Qt冲突
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import glob
import ast
import re


DATA_DIR = "./x-24-us-election"
OUTPUT_PREFIX = "x_24_us_election"
SKIP_DOWNLOAD = False

PARTS_CONFIG = {
    1: (1, 20),
    2: (21, 40),
    3: (41, 60),
    4: (61, 80),
    5: (81, 100),
    6: (101, 120),
    7: (121, 140),
    8: (141, 160),
    9: (161, 180),
    10: (181, 200),
    11: (201, 220),
    12: (221, 240),
    13: (241, 260),
    14: (261, 280),
    15: (281, 300),
    16: (301, 320),
    17: (321, 340),
    18: (341, 360),
    19: (361, 380),
    20: (381, 400),
    21: (401, 420),
    22: (421, 440),
}

USE_TIME_FILTER = True
START_DATE = '2024-06-08'   
END_DATE = '2024-06-15'

def download_data(parts_config):
    if not os.path.exists(DATA_DIR):
        print("\nDownloading data...")
        subprocess.run([
            'git', 'clone', '--filter=blob:none', '--sparse',
            'https://github.com/sinking8/x-24-us-election.git', DATA_DIR
        ], check=True)
        os.chdir(DATA_DIR)
        subprocess.run(['git', 'sparse-checkout', 'init', '--cone'], check=True)
        os.chdir('..')

    os.chdir(DATA_DIR)
    for part_num in parts_config.keys():
        subprocess.run(['git', 'sparse-checkout', 'add', f'part_{part_num}'], check=False)
    os.chdir('..')
    print("\nData downloaded.")


def find_data_files(parts_config):
    all_files = []
    for part_num, (chunk_start, chunk_end) in parts_config.items():
        part_dir = os.path.join(DATA_DIR, f'part_{part_num}')
        if not os.path.exists(part_dir):
            print(f"part_{part_num} doesn't exist.")
            continue

        print(f"\npart_{part_num}:")
        for i in range(chunk_start, chunk_end + 1):
            matches = glob.glob(os.path.join(part_dir, f"*chunk_{i}.csv.gz"))
            if matches:
                all_files.append(matches[0])
                size_mb = os.path.getsize(matches[0]) / 1024 / 1024
                print(f"chunk_{i}: {os.path.basename(matches[0])} ({size_mb:.1f}MB)")
            else:
                print(f"chunk_{i} doesn't exist.")
    print(f"Find {len(all_files)} data files.")
    return all_files

def load_and_filter_data(files):
    dfs = []
    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {os.path.basename(f)}", end=" ... ")
        try:
            df = pd.read_csv(f, compression='gzip', usecols=['id', 'user', 'retweetedUserID', 'epoch'])
            dfs.append(df)
        except Exception as e:
            print(f"{e}")

    df = pd.concat(dfs, ignore_index=True)

    df = df.drop_duplicates(subset=['id'], keep='first')
    
    if USE_TIME_FILTER:
        df['datetime'] = pd.to_datetime(df['epoch'], unit='s', errors='coerce')
        df = df[df['datetime'].notna()]
        mask = (df['datetime'] >= pd.to_datetime(START_DATE)) & \
               (df['datetime'] < pd.to_datetime(END_DATE))
        df = df[mask]
     
    print(f"\n{len(df):,} tweets")
    
    retweets = df[df['retweetedUserID'].notna()].copy()
    print(f"\nRetweets: {len(retweets):,} ({len(retweets)/len(df)*100:.1f}%)")

    USER_ID_RE = re.compile(r"'id_str':\s*'(\d+)'|\'id\':\s*(\d+)")

    def parse_user_id(val):
        if isinstance(val, dict):
            return str(val.get('id_str') or val.get('id'))
        if isinstance(val, str):
            match = USER_ID_RE.search(val)
            if match:
                return match.group(1) or match.group(2)
        return None

    df['user_id'] = df['user'].map(parse_user_id)
    retweets['user_id'] = retweets['user'].map(parse_user_id)

    retweets['retweetedUserID'] = (
        retweets['retweetedUserID']
        .apply(lambda x: str(int(x)) if pd.notna(x) else None)
    )


    return df, retweets


def build_network(df, retweets):
    G = nx.MultiDiGraph()
    
    all_users = set(df['user_id'].dropna().astype(str)) | \
        set(retweets['retweetedUserID'].dropna().astype(str))
    G.add_nodes_from(all_users)

    edges_df = retweets[
        retweets['user_id'].notna() &
        retweets['retweetedUserID'].notna()
    ]

    edges = [
        (u, v, {'timestamp': t})
        for u, v, t in zip(
            edges_df['user_id'].astype(str),
            edges_df['retweetedUserID'].astype(str),
            edges_df['epoch']
        )
    ]

    G.add_edges_from(edges)

    print(f"Nodes: {G.number_of_nodes():,}")
    print(f"Edges: {G.number_of_edges():,}")
    print(f"Isolated nodes: {nx.number_of_isolates(G):,}")
    
    return G
# ----------------------------------------------------------
def analyze_network(G):
    """
    Performs structural analysis on the constructed network.
    Compatible with both DiGraph and MultiDiGraph.
    """
    sep_line = "-" * 60 
    
    print("\n" + sep_line)
    print("PHASE 2: NETWORK STRUCTURE ANALYSIS (Topology)")
    print(sep_line)


    # preprocessing
    if G.is_multigraph():
        G_simple = nx.DiGraph(G) 
    else:
        G_simple = G

    # 1. Basic Overview
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    density = nx.density(G_simple)
    reciprocity = nx.reciprocity(G_simple)
    
    print(f"[1] Network Overview")
    print(f"    - Nodes (Users)        : {num_nodes:,}")
    print(f"    - Edges (Interactions) : {num_edges:,}")
    print(f"    - Network Density      : {density:.6e}")
    print(f"    - Reciprocity          : {reciprocity:.4f}")

    # 2. Degree Analysis
    print(f"\n[2] Degree Analysis")
    # degree = Total Retweets
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]

    avg_in = np.mean(in_degrees)
    max_in = np.max(in_degrees)
    avg_out = np.mean(out_degrees)
    max_out = np.max(out_degrees)

    print(f"    - Avg In-Degree (Influence) : {avg_in:.2f} (Max: {max_in:,})")
    print(f"    - Avg Out-Degree (Activity) : {avg_out:.2f} (Max: {max_out:,})")

    try:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        bins_in = np.logspace(np.log10(1), np.log10(max_in+1), 50)
        plt.hist(in_degrees, bins=bins_in, alpha=0.7, color='#1f77b4', log=True)
        plt.xscale('log')
        plt.yscale('log')
        plt.title('In-Degree Distribution (Influence)')
        plt.xlabel('Degree (k)')
        plt.ylabel('Frequency P(k)')
        
        plt.subplot(1, 2, 2)
        bins_out = np.logspace(np.log10(1), np.log10(max_out+1), 50)
        plt.hist(out_degrees, bins=bins_out, alpha=0.7, color='#ff7f0e', log=True)
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Out-Degree Distribution (Activity)')
        plt.xlabel('Degree (k)')
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_PREFIX}_degree_dist.png")
        print(f"    -> Plot saved: {OUTPUT_PREFIX}_degree_dist.png")
        plt.close()
    except Exception as e:
        print(f"    -> Plotting failed: {e}")

    # 3. Path Length Analysis

    print(f"\n[3] Path Length Analysis")
    if num_nodes > 0:
        scc = list(nx.strongly_connected_components(G_simple))
        largest_scc = max(scc, key=len)
        
        if 1 < len(largest_scc) < 3000:
            sub_g = G_simple.subgraph(largest_scc)
            avg_path = nx.average_shortest_path_length(sub_g)
            print(f"    - Avg Path Length : {avg_path:.4f} (Calculated on largest connected subgraph)")
        else:
            print(f"    - Avg Path Length : Skipped (Graph is too fragmented or too large)")


    # 4. Clustering Analysis
    print(f"\n[4] Clustering Analysis")
    try:
        # G_simple is essential
        transitivity = nx.transitivity(G_simple)
        print(f"    - Transitivity (Global) : {transitivity:.6f}")
    except Exception as e:
        print(f"    - Transitivity calculation failed: {e}")


    # 5. Assortativity Analysis
    
    print(f"\n[5] Assortativity Analysis")
    try:
        # Assortativity support MultiGraph
        r = nx.degree_assortativity_coefficient(G)
        print(f"    - Degree Assortativity  : {r:.4f}")
    except:
        print("    - Assortativity calculation failed.")

    
    # 6. Centrality Analysis (Top 10) 
    print(f"\n[6] Centrality Analysis (Top 10)")
    
    top_in_degree = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:10]
    print("\n    [Most Retweeted Users (In-Degree)]")
    for i, (user, deg) in enumerate(top_in_degree, 1):
        print(f"    {i:2d}. User {user:<20} : {deg}")

    print("\n    [Calculating PageRank...]")
    try:
        pr = nx.pagerank(G, alpha=0.85)
        top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]
        print("    [Top Authority Users (PageRank)]")
        for i, (user, score) in enumerate(top_pr, 1):
            print(f"    {i:2d}. User {user:<20} : {score:.6f}")
            
        with open(f"{OUTPUT_PREFIX}_centrality_results.txt", "w") as f:
            f.write("Rank,Type,User_ID,Score\n")
            for i, (u, s) in enumerate(top_pr, 1):
                f.write(f"{i},PageRank,{u},{s}\n")
            for i, (u, s) in enumerate(top_in_degree, 1):
                f.write(f"{i},InDegree,{u},{s}\n")
        print(f"    -> Centrality results saved to {OUTPUT_PREFIX}_centrality_results.txt")
        
    except Exception as e:
        print(f"    -> PageRank failed: {e}")

    print("\n" + sep_line)

#------------------------------------------------------------------

def export_network(G, parts_config):
    if len(parts_config) == 1:
        part_num = list(parts_config.keys())[0]
        start, end = parts_config[part_num]
        suffix = f"part{part_num}_chunk{start}-{end}"
    else:
        suffix = "multipart_" + "_".join([f"p{p}" for p in sorted(parts_config.keys())])

    gexf_path = f"{OUTPUT_PREFIX}_{suffix}.gexf"
    nx.write_gexf(G, gexf_path)
    print(f"Saved to {gexf_path}")


def visualize_edge_timestamps(G):
    print("\nVisualizing edge timestamps...")
    timestamps = []

    for u, v, data in G.edges(data=True):
        if 'timestamp' in data:
            timestamps.append(data['timestamp'])

    if not timestamps:
        print("No timestamps found in edges.")
        return

    dt_times = pd.to_datetime(timestamps, unit='s')

    start_time = dt_times.min().floor('h')
    end_time = dt_times.max().ceil('h')

    hourly_bins = pd.date_range(start=start_time, end=end_time, freq='h')

    plt.figure(figsize=(14, 6))
    plt.hist(dt_times, bins=hourly_bins, color='skyblue', edgecolor='black')
    plt.title('Distribution of Retweet Timestamps (Hourly)')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_file = "edge_time_distribution.png"
    plt.savefig(output_file)
    print(f"Saved timestamp distribution to {output_file}")
    plt.close()


def main():
    start_time = time.time()

    print(f"\nConfig: {len(PARTS_CONFIG)} part(s)")
    for part_num, (start, end) in PARTS_CONFIG.items():
        print(f"Part {part_num}: chunk {start}-{end}")
    print(f"Time: {START_DATE} to {END_DATE}" if USE_TIME_FILTER else "  Time: No filter")

    if not SKIP_DOWNLOAD:
        download_data(PARTS_CONFIG)

    files = find_data_files(PARTS_CONFIG)
    if not files:
        print("No data files found.")
        return
    
    df, retweets = load_and_filter_data(files)
    if len(retweets) == 0:
        print("No retweet data.")
        return
    
    G = build_network(df, retweets)
    export_network(G, PARTS_CONFIG)
    analyze_network(G)

    visualize_edge_timestamps(G)

    elapsed = time.time() - start_time
    print(f"Done. Time: {elapsed/60:.1f} min")

if __name__ == "__main__":

    main()


