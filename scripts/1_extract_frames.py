import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_folder, frame_rate=2):
    """
    Extracts frames from a video file at a specified frame rate.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Directory to save the extracted frames.
        frame_rate (int): Extract one frame every 'frame_rate' seconds.
    """
    video_path = Path(video_path)
    output_folder = Path(output_folder)
    
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        return

    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate) if frame_rate > 0 else 1
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        if frame_count % frame_interval == 0:
            frame_filename = output_folder / f"{saved_count:05d}.png"
            cv2.imwrite(str(frame_filename), frame)
            saved_count += 1
        
        frame_count += 1

    cap.release()
    print(f"Successfully extracted {saved_count} frames to {output_folder}")

if __name__ == '__main__':
    VIDEO_FILE = 'data/video.mp4'
    FRAME_OUTPUT_DIR = 'data/frames'
    FRAMES_PER_SECOND = 1
    extract_frames(VIDEO_FILE, FRAME_OUTPUT_DIR, FRAMES_PER_SECOND)