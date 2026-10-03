"""
scripts/create_test_videos.py
Creates sample video files (fire_test.mp4 and smoke_test.mp4) for offline testing
and verification of the video capture and detection pipeline.
"""

import cv2
import numpy as np
from pathlib import Path

def create_sample_videos():
    base_dir = Path(__file__).resolve().parent.parent
    videos_dir = base_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    width, height = 640, 480
    fps = 30
    duration_sec = 2
    total_frames = fps * duration_sec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 1. Fire Test Video
    fire_video_path = videos_dir / "fire_test.mp4"
    out_fire = cv2.VideoWriter(str(fire_video_path), fourcc, fps, (width, height))
    print(f"Generating {fire_video_path.name}...")

    for i in range(total_frames):
        # Dark room background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (20, 20, 25)

        # Dynamic flickering fire simulation
        center_x = width // 2 + int(10 * np.sin(i * 0.5))
        center_y = height // 2 + 50
        radius_x = 70 + int(15 * np.cos(i * 0.8))
        radius_y = 110 + int(20 * np.sin(i * 0.6))

        # Outer red glow
        cv2.ellipse(frame, (center_x, center_y), (radius_x + 30, radius_y + 30), 0, 0, 360, (0, 40, 200), -1)
        # Inner orange flame
        cv2.ellipse(frame, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, (0, 140, 255), -1)
        # Core yellow flame
        cv2.ellipse(frame, (center_x, center_y + 20), (radius_x // 2, radius_y // 2), 0, 0, 360, (30, 230, 255), -1)

        cv2.putText(frame, f"TEST FEED: FIRE SIMULATION [Frame {i+1}/{total_frames}]", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out_fire.write(frame)

    out_fire.release()
    print(f"Created {fire_video_path} successfully ({total_frames} frames).")

    # 2. Smoke Test Video
    smoke_video_path = videos_dir / "smoke_test.mp4"
    out_smoke = cv2.VideoWriter(str(smoke_video_path), fourcc, fps, (width, height))
    print(f"Generating {smoke_video_path.name}...")

    for i in range(total_frames):
        # Industrial background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (35, 35, 40)

        # Billowing smoke simulation (rising grey plumes)
        y_offset = int((i / total_frames) * 120)
        center_x = width // 2 + int(30 * np.sin(i * 0.3))
        center_y = height - 120 - y_offset
        radius = 80 + i * 2

        # Semi-transparent layers of grey
        overlay = frame.copy()
        cv2.circle(overlay, (center_x, center_y), radius, (160, 160, 160), -1)
        cv2.circle(overlay, (center_x + 40, center_y - 30), radius - 15, (190, 190, 190), -1)
        cv2.circle(overlay, (center_x - 30, center_y - 20), radius - 20, (140, 140, 140), -1)

        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, f"TEST FEED: SMOKE SIMULATION [Frame {i+1}/{total_frames}]", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out_smoke.write(frame)

    out_smoke.release()
    print(f"Created {smoke_video_path} successfully ({total_frames} frames).")

if __name__ == "__main__":
    create_sample_videos()
