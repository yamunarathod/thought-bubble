from ultralytics import YOLO
import glob
import os
import cv2
import pandas as pd
import argparse
import random

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

# Draws a simple oval-shaped thought bubble and returns its bounding box

def draw_thought_bubble(img, x, y, text, scale=1.0):
    font_scale = 0.6 * scale
    font_thickness = max(1, int(2 * scale))
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

    padding_x, padding_y = int(20 * scale), int(15 * scale)
    bubble_w = text_w + padding_x * 2
    bubble_h = text_h + padding_y * 2

    center = (x, y - bubble_h // 2)
    axes = (bubble_w // 2, bubble_h // 2)

    # Draw white oval bubble
    cv2.ellipse(img, center, axes, 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, center, axes, 0, 0, 360, (0, 0, 0), 1)

    # Draw tail
    cv2.circle(img, (x - int(15 * scale), y + int(10 * scale)), int(8 * scale), (255, 255, 255), -1)
    cv2.circle(img, (x - int(5 * scale), y + int(18 * scale)), int(5 * scale), (255, 255, 255), -1)

    # Draw text
    text_org = (x - text_w // 2, y + text_h // 2 - bubble_h)
    cv2.putText(img, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

    # Return top-left and bottom-right corners for overlap detection
    top_left = (center[0] - axes[0], center[1] - axes[1])
    bottom_right = (center[0] + axes[0], center[1] + axes[1])
    return top_left, bottom_right

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    model = YOLO(args.model)

    if args.mode == 'detect':
        df = pd.DataFrame(columns=['name', 'xmin', 'ymin', 'xmax', 'ymax'])

        source = 0 if args.source == 'camera' else args.source
        results = model(source, stream=True)

        head_thoughts = {}
        thought_texts = [
            "Hmm... thinking of AI",
            "What's for lunch?",
            "Hope nobody saw that",
            "Did I lock the door?",
            "Existence is wild.",
            "Brain loading...",
            "Where’s the pizza?",
            "Weekend mode: ON",
            "Staring into nothing",
            "Need coffee ☕"
        ]

        for result in results:
            frame = result.orig_img.copy()
            boxes = result.boxes

            if boxes is not None:
                box_info = []
                for i, box in enumerate(boxes):
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = b
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Add minimum area threshold (in pixels)
                    min_detection_area = 3000  # Adjust this value as needed
                    if area >= min_detection_area:
                        box_info.append((area, i, (x1, y1, x2, y2)))

                box_info.sort(reverse=True)
                max_bubbles = len(box_info)
                used_bubble_boxes = []

                for idx, (area, i, (x1, y1, x2, y2)) in enumerate(box_info):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    center_x = (x1 + x2) // 2
                    height = y2 - y1
                    top_y = y1 - 20

                    # Jitter-resistant ID
                    rounded_x = round(center_x / 50) * 50
                    rounded_y = round(top_y / 50) * 50
                    head_id = f"{rounded_x}_{rounded_y}"

                    # Scale bubble by size
                    min_area, max_area = 1000, 30000
                    scale = min(1.2, max(0.6, (area - min_area) / (max_area - min_area)))

                    # Dynamic Y-position (taller person = higher bubble)
                    bubble_y = y1 - int(height * 0.6)

                    # Prevent overlap
                    for _ in range(10):
                        bubble_top, bubble_bottom = bubble_y - int(80 * scale), bubble_y
                        overlap = False
                        for (tl, br) in used_bubble_boxes:
                            if (abs(center_x) > tl[0] and abs(center_x) < br[0]) and not (bubble_bottom < tl[1] or bubble_top > br[1]):
                                overlap = True
                                bubble_y -= int(20 * scale)
                                break
                        if not overlap:
                            break

                    if idx < max_bubbles:
                        if head_id not in head_thoughts:
                            head_thoughts[head_id] = random.choice(thought_texts)
                        thought = head_thoughts[head_id]
                        top_left, bottom_right = draw_thought_bubble(frame, center_x, bubble_y, thought, scale)
                        used_bubble_boxes.append((top_left, bottom_right))

                    name = 'camera' if args.source == 'camera' else result.path.split("/")[-1]
                    xyxyn = boxes[i].xyxyn[0].cpu().numpy()
                    df.loc[len(df)] = [name] + list(xyxyn)

            if args.source == 'camera':
                cv2.imshow("Head Detection with Thought Bubbles", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        df.to_csv(args.output, index=False)
        if args.source == 'camera':
            cv2.destroyAllWindows()

    else:
        print("\u274c Please enter a valid mode: detect")
