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

# 🧠 Draw resizable thought bubble with text
def draw_thought_bubble(img, x, y, text):
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    padding_x, padding_y = 20, 10
    bubble_w = text_w + padding_x * 2
    bubble_h = text_h + padding_y * 2

    # Main bubble
    top_left = (x - bubble_w // 2, y - bubble_h - 30)
    bottom_right = (x + bubble_w // 2, y - 30)
    cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), -1)
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), 1)

    # Tail
    cv2.circle(img, (x - 15, y - 15), 8, (255, 255, 255), -1)
    cv2.circle(img, (x - 5, y - 8), 5, (255, 255, 255), -1)

    # Text inside
    text_org = (top_left[0] + padding_x, bottom_right[1] - padding_y - 5)
    cv2.putText(img, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    model = YOLO(args.model)

    if args.mode == 'detect':
        df = pd.DataFrame(columns=['name', 'xmin', 'ymin', 'xmax', 'ymax'])

        source = 0 if args.source == 'camera' else args.source
        results = model(source, stream=True)

        # Thought bubble memory
        head_thoughts = {}
        thought_texts = [
            "Hmm... thinking of AI",
            "What's for lunch?",
            "Hope nobody saw that",
            "Did I lock the door?",
            "Existence is wild.",
            "Brain loading...",
            "Weekend plans loading",
            "Where's the pizza?",
            "Thinking about deadlines",
            "Just a floating thought"
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
                    box_info.append((area, i, (x1, y1, x2, y2)))

                # Sort by size (bigger = closer)
                box_info.sort(reverse=True)

                max_bubbles = 2  # max heads to show thoughts
                for idx, (area, i, (x1, y1, x2, y2)) in enumerate(box_info):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Use center point as lightweight ID
                    center_x = (x1 + x2) // 2
                    top_y = y1 - 20
                    rounded_x = round(center_x / 50) * 50
                    rounded_y = round(top_y / 50) * 50
                    head_id = f"{rounded_x}_{rounded_y}"


                    if idx < max_bubbles:
                        # Persist or assign new thought
                        if head_id not in head_thoughts:
                            head_thoughts[head_id] = random.choice(thought_texts)
                        thought = head_thoughts[head_id]
                        draw_thought_bubble(frame, center_x, top_y, thought)

                    # Save CSV
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
        print("❌ Please enter a valid mode: detect")
