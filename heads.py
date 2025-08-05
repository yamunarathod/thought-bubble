from ultralytics import YOLO
import glob
import os
import cv2
import pandas as pd
import argparse

def get_parser():
    num = len(glob.glob("./output_*.csv"))
    parser = argparse.ArgumentParser(description='Setting parameters of the model.')
    parser.add_argument('--model', type=str, default='medium.pt',
                        help='Path to the model weights')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image folder or "camera" for webcam')
    parser.add_argument('--output', type=str, default="./output_%d.csv" % num,
                        help='Path to the output CSV file')
    parser.add_argument('--mode', type=str, default="detect",
                        help='Mode of the model: detect or track')
    return parser

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
                for box in boxes:
                    # Draw bounding box
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = b
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Save to CSV
                    name = 'camera' if args.source == 'camera' else result.path.split("/")[-1]
                    xyxyn = box.xyxyn[0].cpu().numpy()
                    df.loc[len(df)] = [name] + list(xyxyn)

            if args.source == 'camera':
                cv2.imshow("Head Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        df.to_csv(args.output, index=False)
        if args.source == 'camera':
            cv2.destroyAllWindows()

    elif args.mode == 'track':
        df = pd.DataFrame(columns=['name', 'id', 'xmin', 'ymin', 'xmax', 'ymax'])
        results = model.track(args.source, stream=True, persist=True)

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i in range(boxes.shape[0]):
                if result.boxes.is_track:
                    data = [result.path.split("/")[-1], int(boxes.id[i])] + list(boxes.xyxyn[i])
                    df.loc[len(df)] = data

        df.to_csv(args.output, index=False)

    else:
        print("❌ Please enter a proper mode: detect or track")
