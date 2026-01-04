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

def analyze_network(G):
    
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    
    print(f"In-degree: avg={np.mean(list(in_deg.values())):.2f}, max={max(in_deg.values()):,}")
    print(f"Out-degree: avg={np.mean(list(out_deg.values())):.2f}, max={max(out_deg.values()):,}")

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