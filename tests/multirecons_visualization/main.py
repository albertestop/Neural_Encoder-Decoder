import cv2
import numpy as np
from pathlib import Path
import math
import sys
import os

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

from config import *


def reconstruct_composite_video(trial_save_paths, composite_video_path, fps=30, grid_cols = 4):
    """
    Creates a composite video that displays multiple TIFF videos at the same time in a grid.
    
    Parameters:
      trial_save_paths (list of str): List of directories containing an "optimized_input.tif" file.
      composite_video_path (str): Path for the composite output video (e.g. "composite.mp4").
      fps (int): Frames per second for the output video.
      
    Returns:
      The composite video path.
    """
    videos = []
    
    # Load each TIFF file and convert frames to BGR color space.
    for trial_save_path in trial_save_paths:
        tif_path = Path(f'{trial_save_path}/optimized_input.tif')
        tif_path = parent_dir/tif_path
        if not tif_path.exists():
            raise FileNotFoundError(f"The TIFF file {tif_path} does not exist.")
                
        # imreadmulti returns a tuple: (retval, [list of frames])
        ret, tif_frames = cv2.imreadmulti(str(tif_path))
        
        # Convert grayscale frames to BGR
        bgr_frames = [cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) for frame in tif_frames]
        videos.append(bgr_frames)
    
    # Determine the number of frames to iterate over: use the minimum count among videos
    num_frames = min(len(video) for video in videos)
    
    # Assume all frames have the same dimensions (otherwise, add code to resize)
    frame_height, frame_width = videos[0][0].shape[:2]
    
    # Determine grid layout (auto grid: square-ish)
    num_videos = len(videos)
    grid_rows = math.ceil(num_videos / grid_cols)
    
    # The composite frame size is computed by placing each video frame in the grid.
    composite_width = grid_cols * frame_width
    composite_height = grid_rows * frame_height
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(composite_video_path, fourcc, fps, (composite_width, composite_height))
    
    for i in range(num_frames):
        row_frames = []
        for r in range(grid_rows):
            frames_in_row = []
            for c in range(grid_cols):
                index = r * grid_cols + c
                if index < num_videos:
                    frame = videos[index][i]
                else:
                    # Fill empty grid cells with a black frame
                    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                frames_in_row.append(frame)
            # Concatenate frames horizontally for this row
            row_concat = np.hstack(frames_in_row)
            row_frames.append(row_concat)
        # Concatenate all rows vertically to form the composite frame
        composite_frame = np.vstack(row_frames)
        video_writer.write(composite_frame)
    
    video_writer.release()
    return composite_video_path

# Example usage:

composite_path = reconstruct_composite_video(recons_path, 'composite.mp4')
print(f"Composite video saved to {composite_path}")