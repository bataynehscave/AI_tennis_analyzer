from utils import (read_video, write_video)
from trackers import PlayerTracker
import os
from pathlib import Path


def main():
    #paths
    INPUT_VID_PATH = 'input_video/input_video.mp4'
    OUTPUT_VID_DIR = Path('output_videos')
    PLAYER_MODEL_PATH = 'training/models/yolo11x.pt'
    os.makedirs(OUTPUT_VID_DIR, exist_ok=True)

    #read video
    frames = read_video(INPUT_VID_PATH)

    #track
    player_tracker = PlayerTracker(PLAYER_MODEL_PATH)
    players_res = player_tracker.detect_frames(frames=frames)
    bframes = player_tracker.draw_bboxes(frames=frames, players_detection=players_res)
    #export video
    output_name  = 'output.avi'
    write_video(bframes, OUTPUT_VID_DIR / output_name )


if __name__ == '__main__':
    main()