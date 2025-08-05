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

def draw_thought_bubble(img, x, y, text):
    # Draw white ellipse as thought bubble
    cv2.ellipse(img, (x, y - 30), (60, 30), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (x, y - 50), (45, 20), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (x, y - 10), 8, (255, 255, 255), -1)
    cv2.circle(img, (x, y), 5, (255, 255, 255), -1)

    # Draw text inside bubble
    cv2.putText(img, text, (x - 35, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    model = YOLO(args.model)

    if args.mode == 'detect':
        df = pd.DataFrame(columns=['name', 'xmin', 'ymin', 'xmax', 'ymax'])

        source = 0 if args.source == 'camera' else args.source
        results = model(source, stream=True)

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

                # Sort by area (bigger box = closer head)
                box_info.sort(reverse=True)

                # Draw all boxes, and bubbles for top N closest
                max_bubbles = 2
                thought_texts = ["Hmm...", "Pizza?", "What now?", "Thinking...", "🤖🤔"]

                for idx, (area, i, (x1, y1, x2, y2)) in enumerate(box_info):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Thought bubble for closest people
                    if idx < max_bubbles:
                        center_x = (x1 + x2) // 2
                        top_y = y1 - 20
                        thought = random.choice(thought_texts)
                        draw_thought_bubble(frame, center_x, top_y, thought)

                    # Save to CSV
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
