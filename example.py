import ultralytics
from ultralytics import YOLO
import glob
import os
import cv2
import pandas as pd
import argparse
import random
# Import the new bubble drawing function from our separate module
from bubble_drawer import draw_thought_bubble, wrap_text # Import wrap_text too

def get_parser():
    num = len(glob.glob("./output_*.csv"))
    parser = argparse.ArgumentParser(description='Head detection with YOLOv8')
    parser.add_argument('--model', type=str, default='medium.pt',
                        help='Path to the YOLO model')
    parser.add_argument('--source', type=str, required=True,
                        help='Image folder or "camera" for webcam')
    parser.add_argument('--output', type=str, default=f"./output_{num}.csv",
                        help='CSV output path')
    parser.add_argument('--mode', type=str, default='detect',
                        help='Mode: detect or track')
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    model = YOLO(args.model)
    # Move model to GPU if available
    try:
        import torch
        if torch.cuda.is_available():
            model = model.to('cuda')
            print("\u2705 YOLO model moved to GPU (cuda)")
        else:
            print("\u26A0\uFE0F CUDA not available, running on CPU")
    except ImportError:
        print("\u26A0\uFE0F torch not installed, cannot check for GPU. Running on CPU.")

    if args.mode == 'detect':
        df = pd.DataFrame(columns=['name', 'xmin', 'ymin', 'xmax', 'ymax'])
        source = 0 if args.source == 'camera' else args.source
        # Set camera FPS if using webcam
        if args.source == 'camera':
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FPS, 30)
            del cap
        import time
        last_time = time.time()
        frame_count = 0
        results = model(source, stream=True)
        head_thoughts = {}
        # Updated thought_texts list
        thought_texts = [
            "Wow I look great!",
            "I wonder what’s for lunch today.",
            "I can’t wait to invest in Cisco’s AI tech!",
            "AI-Ready Data Centers are my favorite kind of data centers!",
            "There is no defense like Cisco AI-Defense!",
            "Mirror, mirror on the wall, who’s the smartest AI of all? (My bet’s on Cisco AI)",
            "I am reimagining my data center for the AI era, with Cisco",
            "Power, speed, and security—while balancing costs? That’s Cisco AI-Ready Datacenters",
            "Scaling for AI fast? I count on Cisco",
            "For built-in security and resilience, I choose Cisco.",
            "Evolving business needs? I trust Cisco to keep up.",
            "Complexity is everywhere. Cisco keeps it simple for me.",
            "I hope there’s free Wi-Fi…and snacks.",
            "If I nod, will they think I understand everything?",
            "I wonder if anyone will notice if I sneak out for a nap.",
            "Maybe the AI can network for me while I find snacks",
            "AI brings new risks, but with Cisco AI Defense, I feel covered from end to end",
            "With so many third-party AI apps popping up, I’m glad Cisco can spot them before I do.",
            "I want to innovate with AI, not worry about security—and Cisco makes that possible",
            "I’d rather prevent a data leak than clean one up. Cisco AI Defense gets it.",
            "Real-time guardrails for AI? That’s peace of mind, Thank You Cisco.",
            "With Cisco, I get to focus on AI innovation, not AI anxiety.",
            "Should I ask a question, or just nod like I already know the answer?",
            "How many cups of coffee is too many at a tech event?",
            "I’m here for the innovation, but I’ll stay for the pastries."
        ]
        # Set OpenCV window to fullscreen if using camera
        if args.source == 'camera':
            window_name = "Head Detection with Thought Bubbles"
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # --- Simple ID-based Centroid Tracker ---
        max_distance = 60  # pixels, adjust as needed
        max_disappeared = 15  # frames to wait before removing a lost person
        next_person_id = 0
        # tracked_persons: person_id: {'centroid': (x, y), 'disappeared': 0, 'raw_centroid': (x, y), 'bubble_y': int, 'size_cat': 'near'/'far'}
        tracked_persons = {}
        id_to_bubble = {}     # person_id: bubble text
        smoothing_alpha = 0.2  # 0 < alpha <= 1, lower = more smoothing
        near_threshold = 8000 # Define near_threshold here (pixels)
        scale_far = 0.7  # Use this for all bubbles

        def match_or_new(cx, cy, tracked_persons, max_distance):
            # Try to match to existing person by centroid distance
            for pid, info in tracked_persons.items():
                tx, ty = info['raw_centroid'] if 'raw_centroid' in info else info['centroid']
                if ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5 < max_distance:
                    return pid
            return None
        # --- End ID-based Tracker ---

        for result in results:
            frame = result.orig_img.copy()
            frame_count += 1
            if frame_count % 10 == 0:
                now = time.time()
                fps = 10 / (now - last_time)
                print(f"[INFO] Approx FPS: {fps:.2f}")
                last_time = now
            boxes = result.boxes
            # Mark all persons as disappeared by default
            for pid in tracked_persons:
                tracked_persons[pid]['disappeared'] += 1
            if boxes is not None:
                box_info = []
                matched_ids = set()
                for i, box in enumerate(boxes):
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = b
                    area = (x2 - x1) * (y2 - y1)

                    # Add minimum area threshold (in pixels)
                    min_detection_area = 3000  # Adjust this value as needed
                    if area >= min_detection_area:
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        height = y2 - y1
                        # Compute target bubble_y for this frame
                        target_bubble_y = y1 - int(height * 0.2)
                        # Try to match to existing person
                        pid = match_or_new(center_x, center_y, tracked_persons, max_distance)
                        if pid is not None:
                            prev_cx, prev_cy = tracked_persons[pid]['centroid']
                            smoothed_cx = int(smoothing_alpha * center_x + (1 - smoothing_alpha) * prev_cx)
                            smoothed_cy = int(smoothing_alpha * center_y + (1 - smoothing_alpha) * prev_cy)
                            tracked_persons[pid]['centroid'] = (smoothed_cx, smoothed_cy)
                            tracked_persons[pid]['raw_centroid'] = (center_x, center_y)
                            # Smooth bubble_y
                            prev_bubble_y = tracked_persons[pid].get('bubble_y', target_bubble_y)
                            smoothed_bubble_y = int(smoothing_alpha * target_bubble_y + (1 - smoothing_alpha) * prev_bubble_y)
                            tracked_persons[pid]['bubble_y'] = smoothed_bubble_y
                            tracked_persons[pid]['disappeared'] = 0
                            # Update size category only if crossing threshold
                            prev_cat = tracked_persons[pid]['size_cat']
                            if prev_cat == 'far' and area >= near_threshold:
                                tracked_persons[pid]['size_cat'] = 'near'
                            elif prev_cat == 'near' and area < near_threshold:
                                tracked_persons[pid]['size_cat'] = 'far'
                        else:
                            # New person
                            size_cat = 'near' if area >= near_threshold else 'far'
                            pid = next_person_id
                            tracked_persons[pid] = {
                                'centroid': (center_x, center_y),
                                'raw_centroid': (center_x, center_y),
                                'bubble_y': target_bubble_y,
                                'disappeared': 0,
                                'size_cat': size_cat
                            }
                            next_person_id += 1
                        matched_ids.add(pid)
                        # Assign bubble if not already
                        if pid not in id_to_bubble:
                            id_to_bubble[pid] = random.choice(thought_texts)
                        # All bubbles use the same size (far)
                        is_near = 0
                        box_info.append((is_near, area, i, (x1, y1, x2, y2), pid))
                # Remove persons not matched for max_disappeared frames
                to_remove = [pid for pid, info in tracked_persons.items() if info['disappeared'] > max_disappeared]
                for pid in to_remove:
                    tracked_persons.pop(pid)
                    id_to_bubble.pop(pid, None)
                # Sort by size category first (near ones first), then by area
                box_info.sort(key=lambda x: (x[0], x[1]), reverse=True)
                max_bubbles = len(box_info)
                used_bubble_boxes = []
                # In the loop where bubbles are drawn
                for idx, (is_near, area, i, (x1, y1, x2, y2), pid) in enumerate(box_info):
                    # Use smoothed centroid for bubble position
                    center_x, center_y = tracked_persons[pid]['centroid']
                    # Use smoothed bubble_y for vertical position
                    bubble_y = tracked_persons[pid]['bubble_y']
                    # All bubbles use the same scale (far)
                    scale = scale_far

                    # --- Updated current_axes calculation for overlap detection ---
                    estimated_font_scale = 0.6 * scale
                    estimated_font_thickness = max(1, int(2 * scale))
                    estimated_max_content_width = int(220 * scale)
                    dummy_long_text = "This is a very long dummy text to estimate the height of a wrapped bubble."
                    _, estimated_total_wrapped_text_height, _ = wrap_text(
                        dummy_long_text, cv2.FONT_HERSHEY_DUPLEX, estimated_font_scale,
                        estimated_font_thickness, estimated_max_content_width
                    )
                    dummy_padding_x, dummy_padding_y = int(35 * scale), int(25 * scale)
                    dummy_bubble_w = estimated_max_content_width + dummy_padding_x * 2
                    dummy_bubble_h = estimated_total_wrapped_text_height + dummy_padding_y * 2
                    current_axes = (dummy_bubble_w // 2, dummy_bubble_h // 2)
                    # --- End of updated current_axes calculation ---

                    if idx < max_bubbles:
                        thought = id_to_bubble[pid]
                        top_left, bottom_right, final_axes = draw_thought_bubble(frame, center_x, bubble_y, thought, scale)
                        used_bubble_boxes.append((top_left, bottom_right, final_axes))
                    name = 'camera' if args.source == 'camera' else result.path.split("/")[-1]
                    xyxyn = boxes[i].xyxyn[0].cpu().numpy()
                    df.loc[len(df)] = [name] + list(xyxyn)
            if args.source == 'camera':
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        df.to_csv(args.output, index=False)
        if args.source == 'camera':
            cv2.destroyAllWindows()
    else:
        print("\u274c Please enter a valid mode: detect")
