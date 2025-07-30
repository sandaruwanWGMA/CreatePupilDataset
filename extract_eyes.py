# ───────────────────────── extract_eyes.py ─────────────────────────
"""
$ python extract_eyes.py --video my_clip.mp4 --output ./eye_crops --size 256
Result:
eye_crops/my_clip/left/  000000.jpg …
                          000001.jpg …
eye_crops/my_clip/right/ 000000.jpg …
                          000001.jpg …

$ python extract_eyes.py --folder ./videos --output ./eye_crops --size 256
Result:
eye_crops/video1/left/  000000.jpg …
                          000001.jpg …
eye_crops/video1/right/ 000000.jpg …
                          000001.jpg …
eye_crops/video2/left/  000000.jpg …
                          000001.jpg …
eye_crops/video2/right/ 000000.jpg …
                          000001.jpg …
"""
import cv2
import os
import argparse
import mediapipe as mp
import numpy as np
from pathlib import Path

# ❶ Landmark indices for both eyes (MediaPipe Face Mesh, 478-pt model)
LEFT_EYE_IDX  = [ 33,  7, 163, 144, 145, 153, 154, 155,
                 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249,
                 263, 466, 388, 387, 386, 385, 384, 398]

# ❷ Helper: Get rotated eye crop --------------------------------------------
def get_rotated_eye_crop(frame, landmarks, indices, crop_size):
    """
    Return a rotated eye crop (crop_size x crop_size) aligned to the eye orientation.
    """
    h, w = frame.shape[:2]

    # Get eye center
    cx = int(sum(landmarks[i].x for i in indices) / len(indices) * w)
    cy = int(sum(landmarks[i].y for i in indices) / len(indices) * h)

    # Estimate angle between outer and inner eye corners
    p1 = landmarks[indices[0]]   # outer corner
    p2 = landmarks[indices[8]]   # inner corner
    dx = (p2.x - p1.x) * w
    dy = (p2.y - p1.y) * h
    angle = np.degrees(np.arctan2(dy, dx))

    # Rotate image around eye center
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)

    # Crop centered square
    x1 = max(cx - crop_size // 2, 0)
    y1 = max(cy - crop_size // 2, 0)
    x2 = min(x1 + crop_size, w)
    y2 = min(y1 + crop_size, h)

    crop = rotated[y1:y2, x1:x2]

    # Pad if crop smaller than expected
    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        crop = cv2.copyMakeBorder(
            crop,
            top=0, bottom=crop_size - crop.shape[0],
            left=0, right=crop_size - crop.shape[1],
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

    return crop

# ❸ Core ---------------------------------------------------------------------
def process_video(video_path, out_root, crop_size):
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    # Prepare output dirs
    left_dir  = Path(out_root) / video_path.stem / "left"
    right_dir = Path(out_root) / video_path.stem / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    mp_face_mesh = mp.solutions.face_mesh
    mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,   # Iris landmarks included
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_idx, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        results = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # --- Left eye ---
            crop_l = get_rotated_eye_crop(frame, landmarks, LEFT_EYE_IDX, crop_size)
            crop_l_gray = cv2.cvtColor(crop_l, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            cv2.imwrite(str(left_dir / f"{frame_idx:06d}.jpg"), crop_l_gray)

            # --- Right eye ---
            crop_r = get_rotated_eye_crop(frame, landmarks, RIGHT_EYE_IDX, crop_size)
            crop_r_gray = cv2.cvtColor(crop_r, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            cv2.imwrite(str(right_dir / f"{frame_idx:06d}.jpg"), crop_r_gray)

            saved += 2
        frame_idx += 1

    cap.release()
    mesh.close()
    print(f"✓ Done. {saved} eye crops saved to “{out_root}/{video_path.stem}/left|right”")

# ❹ CLI ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract rotated eye crops using MediaPipe")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="single video file to process")
    group.add_argument("--folder", help="process ALL videos (recursively) in this folder")
    parser.add_argument("--output", required=True, help="folder to store crops")
    parser.add_argument("--size",   type=int, default=256, help="output crop size (square)")
    args = parser.parse_args()

    # Decide between single video or folder mode
    if args.folder:
        root = Path(args.folder)
        video_ext = {'.mp4', '.mov', '.avi', '.mkv', '.mpg', '.mpeg'}
        videos = [p for p in root.rglob('*') if p.suffix.lower() in video_ext]
        if not videos:
            raise FileNotFoundError(f"No video files found under {root}")
        for vp in videos:
            process_video(vp, args.output, args.size)
    else:
        process_video(args.video, args.output, args.size)


'''

python extract_eyes.py \
--video "data/raw/WhatsApp_Video.mp4" \
--output "data/interim/eye_crops" \
--size 256

python extract_eyes.py \
--folder "data/raw/videos" \
--output "data/interim/eye_crops" \
--size 256

'''

'''

python extract_eyes.py \
  --folder data/raw/videos \
  --output data/interim/eye_crops \
  --size 256 \
&& \
python super_resolution/SwinIR/main_test_swinir.py \
  --task gray_dn --noise 15 \
  --model_path model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth \
  --input_dir data/interim/eye_crops \
  --output_dir data/interim/denoised_eye_crops

'''