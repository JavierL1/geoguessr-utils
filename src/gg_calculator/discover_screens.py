# /// script
# dependencies = [
#     "opencv-python",
# ]
# ///

import argparse
import os
import cv2
import sys
import time

def parse_time_in_seconds(hours: int, minutes: int, seconds: int) -> int | None:
    if (
        not isinstance(hours, int)
        or not isinstance(minutes, int)
        or not isinstance(seconds, int)
    ):
        return None
    return hours * 60 * 60 + minutes * 60 + seconds

def parse_args():
    parser = argparse.ArgumentParser(description="Fast discovery of Geoguessr score screens.")
    parser.add_argument("-v", "--video", required=True, help="Path to video file")
    parser.add_argument("-o", "--output", required=True, help="Output directory for saved frames")
    
    parser.add_argument(
        "--start-time", type=int, nargs=3, metavar=("HH", "MM", "SS"), 
        default=[0, 0, 0], help="Start time in hours, minutes, seconds"
    )
    parser.add_argument(
        "--end-time", type=int, nargs=3, metavar=("HH", "MM", "SS"), 
        default=None, help="End time in hours, minutes, seconds"
    )
    
    parser.add_argument("--roi-left", type=int, nargs=4, default=[310, 115, 100, 60], help="Left score ROI")
    parser.add_argument("--roi-right", type=int, nargs=4, default=[1510, 115, 100, 60], help="Right score ROI")
    return parser.parse_args()

def is_score_screen(frame, roi_left, roi_right) -> bool:
    x1, y1, w1, h1 = roi_left
    x2, y2, w2, h2 = roi_right
    
    # Extract ROIs
    left_crop = frame[y1:y1+h1, x1:x1+w1]
    right_crop = frame[y2:y2+h2, x2:x2+w2]
    
    # Bounds Safety Check
    if left_crop.size == 0 or right_crop.size == 0:
        frame_h, frame_w = frame.shape[:2]
        raise ValueError(
            f"\nROI is out of bounds! Video resolution is {frame_w}x{frame_h}.\n"
            f"Left ROI: {roi_left} | Right ROI: {roi_right}\n"
            f"Please adjust your --roi-left and --roi-right arguments."
        )
    
    gray_left = cv2.cvtColor(left_crop, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_crop, cv2.COLOR_BGR2GRAY)
    
    _, thresh_left = cv2.threshold(gray_left, 200, 255, cv2.THRESH_BINARY)
    _, thresh_right = cv2.threshold(gray_right, 200, 255, cv2.THRESH_BINARY)
    
    left_white_ratio = cv2.countNonZero(thresh_left) / (w1 * h1)
    right_white_ratio = cv2.countNonZero(thresh_right) / (w2 * h2)
    
    return left_white_ratio > 0.05 and right_white_ratio > 0.05

def main():
    args = parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Failed to open {args.video}", file=sys.stderr)
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Parse boundaries
    start_seconds = parse_time_in_seconds(*args.start_time) or 0
    if args.end_time:
        end_seconds = parse_time_in_seconds(*args.end_time)
    else:
        end_seconds = total_frames / fps if fps > 0 else 0
        
    start_frame_idx = int(start_seconds * fps)
    end_frame_idx = int(end_seconds * fps) if end_seconds else total_frames
    
    if start_frame_idx >= total_frames:
        print("Error: Start time exceeds video length.", file=sys.stderr)
        sys.exit(1)
    
    # State Machine Variables
    search_step_frames = int(max(1, fps * 1.0))
    collect_step_frames = int(max(1, fps * 0.25))
    
    is_collecting = False
    batch_count = 0
    current_frame_idx = start_frame_idx
    total_frames_to_process = end_frame_idx - start_frame_idx
    
    # Performance tracking
    loop_iterations = 0
    start_time_real = time.time()
    
    print(f"Starting discovery on '{args.video}'...")
    print(f"Processing window: {start_seconds}s to {end_seconds}s (Total frames to span: {total_frames_to_process})")
    
    while current_frame_idx < total_frames and current_frame_idx <= end_frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        detected = is_score_screen(frame, args.roi_left, args.roi_right)
        
        if detected:
            if not is_collecting:
                batch_count += 1
                # Use \n to break out of the carriage-return progress line
                print(f"\n[ {(current_frame_idx/fps):.2f}s ] Score screen detected! Started Batch {batch_count:03d} (0.25s interval)")
                is_collecting = True
            
            out_path = os.path.join(args.output, f"batch_{batch_count:03d}_frame_{current_frame_idx:07d}.jpg")
            cv2.imwrite(out_path, frame)
            
            current_frame_idx += collect_step_frames
        else:
            if is_collecting:
                print(f"\n[ {(current_frame_idx/fps):.2f}s ] Score screen ended. Reverting to 1.0s search interval.")
                is_collecting = False
            
            current_frame_idx += search_step_frames
            
        # Progress Update (Update CLI every 5 reads to avoid flickering)
        loop_iterations += 1
        if loop_iterations % 5 == 0:
            elapsed_real = time.time() - start_time_real
            progress_frames = current_frame_idx - start_frame_idx
            
            if elapsed_real > 0 and progress_frames > 0:
                pct = min(100.0, (progress_frames / total_frames_to_process) * 100)
                read_fps = loop_iterations / elapsed_real
                video_speed = (progress_frames / fps) / elapsed_real
                
                remaining_frames = total_frames_to_process - progress_frames
                eta_sec = remaining_frames / (progress_frames / elapsed_real)
                mins, secs = divmod(int(eta_sec), 60)
                
                # \r overwrites the current terminal line
                sys.stdout.write(
                    f"\rProgress: {pct:5.1f}% | Batches: {batch_count} | Read: {read_fps:4.1f} fps | Speed: {video_speed:4.1f}x | ETA: {mins:02d}:{secs:02d} "
                )
                sys.stdout.flush()

    cap.release()
    print("\n\nDiscovery complete.")

if __name__ == "__main__":
    main()
