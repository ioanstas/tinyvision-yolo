"""
Create GIF from a range of annotated frames.

Usage:
    python scripts/make_gif.py --model p2_baseline --split val --start DJI_0111_2_00_00009 --end DJI_0123_6_00_00295
    python scripts/make_gif.py --model naive_baseline --split val --start DJI_0111_2_00_00009 --end DJI_0123_6_00_00295
"""

import argparse
import cv2
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def make_gif(model, split, start, end, fps=15):
    frames_dir = PROJECT_ROOT / "outputs" / "visualizations" / model / split
    all_files = sorted(frames_dir.glob("*.jpg"))

    if not all_files:
        print(f"ERROR: No frames in {frames_dir}")
        return

    # Filter by range (alphabetical sort works since filenames are consistent)
    start_name = start if start.endswith(".jpg") else start + ".jpg"
    end_name = end if end.endswith(".jpg") else end + ".jpg"
    files = [f for f in all_files if start_name <= f.name <= end_name]

    if not files:
        print(f"ERROR: No frames found between {start} and {end}")
        print(f"  First available: {all_files[0].stem}")
        print(f"  Last available:  {all_files[-1].stem}")
        return

    print(f"Found {len(files)} frames ({files[0].stem} -> {files[-1].stem})")

    frames = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(img_rgb))

    out = PROJECT_ROOT / "outputs" / "visualizations" / "videos"
    out.mkdir(parents=True, exist_ok=True)
    gif_path = out / f"{model}_{start}_to_{end}.gif"

    duration = int(1000 / fps)
    frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"Saved: {gif_path} ({len(frames)} frames, {fps}fps)")


def main():
    parser = argparse.ArgumentParser(description="Create GIF from annotated frames")
    parser.add_argument("--model", type=str, default="p2_baseline")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--start", type=str, required=True, help="First frame name (without .jpg)")
    parser.add_argument("--end", type=str, required=True, help="Last frame name (without .jpg)")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    make_gif(args.model, args.split, args.start, args.end, args.fps)


if __name__ == "__main__":
    main()
