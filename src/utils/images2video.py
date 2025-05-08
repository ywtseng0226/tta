import cv2
import os
import argparse
from glob import glob

def pngs_to_video(image_folder, output_path, fps=10, resize=None):
    images = sorted(glob(os.path.join(image_folder, "*.png")))
    if not images:
        raise ValueError(f"No PNG images found in {image_folder}")

    frame = cv2.imread(images[0])
    if resize:
        frame = cv2.resize(frame, resize)
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in images:
        img = cv2.imread(img_path)
        if resize:
            img = cv2.resize(img, resize)
        out.write(img)

    out.release()
    print(f"✅ Video saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PNGs to video")
    parser.add_argument('--input', type=str, required=True, help='Path to folder containing PNG files')
    parser.add_argument('--output', type=str, required=True, help='Path to save output video file (e.g., out.mp4)')
    parser.add_argument('--fps', type=int, default=2, help='Frames per second (default: 10)')
    parser.add_argument('--resize', nargs=2, type=int, metavar=('width', 'height'), help='Resize frames to width height')

    args = parser.parse_args()
    resize = tuple(args.resize) if args.resize else None

    pngs_to_video(args.input, args.output, fps=args.fps, resize=resize)