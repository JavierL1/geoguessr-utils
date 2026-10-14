# /// script
# dependencies = []
# ///

import argparse
import os
import shutil
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Extract the last frame of each batch into a new folder.")
    parser.add_argument("-i", "--input", required=True, help="Directory containing all the batch frames")
    parser.add_argument("-o", "--output", required=True, help="Directory to save the last frames")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    # Dictionary to hold: batch_id -> list of (frame_idx, filename)
    batches = defaultdict(list)
    
    # 1. Group files by batch ID
    for filename in os.listdir(args.input):
        if not filename.endswith(".jpg") or not filename.startswith("batch_"):
            continue
            
        parts = filename.replace(".jpg", "").split("_")
        try:
            # Filename structure: batch_001_frame_0001234.jpg
            # parts = ['batch', '001', 'frame', '0001234']
            batch_id = int(parts[1])
            frame_idx = int(parts[3])
            batches[batch_id].append((frame_idx, filename))
        except (IndexError, ValueError):
            continue

    if not batches:
        print(f"No correctly formatted batch frames found in '{args.input}'.")
        return

    # 2. Find the last frame per batch and copy it to the new folder
    for batch_id, frames in sorted(batches.items()):
        # Sort by frame index in ascending order
        frames.sort(key=lambda x: x[0])
        last_frame_filename = frames[-1][1]
        
        src_path = os.path.join(args.input, last_frame_filename)
        dst_path = os.path.join(args.output, last_frame_filename)
        
        shutil.copy2(src_path, dst_path)
        print(f"Copied {last_frame_filename}")
        
    print(f"\nSuccessfully extracted {len(batches)} last frames into '{args.output}'.")

if __name__ == "__main__":
    main()
