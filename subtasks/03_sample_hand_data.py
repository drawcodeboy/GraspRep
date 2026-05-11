import argparse
import os
import sys
sys.path.append(os.getcwd())
import random

import numpy as np

from utils.hand_model import HandModel

DEXGRASPNET_ROOT = os.path.join("DexGraspNet", "data", "dexgraspnet")
DEFAULT_SAMPLE_POINTS = 2048

def load_hand_data_files(root_path, limit=None):
    """Load all hand data .npy files from root directory.
    
    Args:
        root_path: path to DexGraspNet/data/dexgraspnet
        limit: maximum number of files to load (None for all)
        
    Returns:
        list of loaded data arrays
    """
    if not os.path.isdir(root_path):
        raise FileNotFoundError(f"Data directory not found: {root_path}")

    npy_file_path = os.path.join(root_path, random.choice(os.listdir(root_path)))
    npy_files = np.load(npy_file_path, allow_pickle=True)

    data_list = []
    for hand_param in npy_files:
        data_list.append(hand_param)

    return data_list

def aggregate_hand_data(data_list, sample_points):
    """Aggregate hand data from multiple objects.
    
    Args:
        data_list: list of loaded data arrays
        sample_points: target number of points to sample/aggregate
        
    Returns:
        numpy array of aggregated points
    """
    if len(data_list) == 0:
        return np.zeros((0,), dtype=object)
    
    # Try to aggregate all data into a single array
    all_data = np.concatenate([np.atleast_1d(d) for d in data_list])
    
    print(f"Aggregated data shape: {all_data.shape}, dtype: {all_data.dtype}")
    
    # If we have more samples than needed, randomly sample
    if len(all_data) > sample_points:
        indices = np.random.choice(len(all_data), sample_points, replace=False)
        all_data = all_data[indices]
        print(f"Sampled down to {sample_points} points")
    
    return all_data


def save_data_npy(data, out_path):
    """Save data as NumPy .npy file."""
    if not out_path.lower().endswith(".npy"):
        raise ValueError("Output path must end with .npy")
    np.save(out_path, data, allow_pickle=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load and aggregate hand grasp data from DexGraspNet"
    )
    parser.add_argument(
        "--root",
        default=DEXGRASPNET_ROOT,
        help="Path to DexGraspNet/data/dexgraspnet directory",
    )
    parser.add_argument(
        "--sample-points",
        type=int,
        default=DEFAULT_SAMPLE_POINTS,
        help="Number of points to sample/aggregate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of files to load (None for all)",
    )
    parser.add_argument(
        "--output",
        default="hand_aggregated.npy",
        help="Output .npy file path",
    )
    return parser.parse_args()


def main():
    hand_model = HandModel(); sys.exit()

    args = parse_args()
    
    print(f"Loading hand data from {args.root}")
    data_list = load_hand_data_files(args.root, limit=args.limit)
    
    if not data_list:
        print("No hand data files found!")
        sys.exit(1)
    
    print(f"\nLoaded {len(data_list)} hand data files")
    
    print(f"\nAggregating hand data (targeting {args.sample_points} points)...")
    aggregated = aggregate_hand_data(data_list, args.sample_points)
    
    print(f"\nSaving aggregated data to {args.output}")
    save_data_npy(aggregated, args.output)
    
    print(f"Done! Saved aggregated hand data with shape {aggregated.shape} to {args.output}")


if __name__ == "__main__":
    main()
