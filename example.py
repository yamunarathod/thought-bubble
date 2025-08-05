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

    if args.mode == 'detect':
        df = pd.DataFrame(columns=['name', 'xmin', 'ymin', 'xmax', 'ymax'])
        source = 0 if args.source == 'camera' else args.source
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
                        # Add a size category flag: 1 for near (large), 0 for far (small)
                        # This is used for scaling the bubble size
                        near_threshold = 8000 # Define near_threshold here
                        is_near = 1 if area >= near_threshold else 0
                        box_info.append((is_near, area, i, (x1, y1, x2, y2)))
                # Sort by size category first (near ones first), then by area
                box_info.sort(key=lambda x: (x[0], x[1]), reverse=True)
                max_bubbles = len(box_info)
                used_bubble_boxes = []
                # In the loop where bubbles are drawn
                for idx, (is_near, area, i, (x1, y1, x2, y2)) in enumerate(box_info):
                    # Remove or comment out this line:
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    center_x = (x1 + x2) // 2
                    height = y2 - y1
                    top_y = y1 - 20
                    # Jitter-resistant ID
                    rounded_x = round(center_x / 50) * 50
                    rounded_y = round(top_y / 50) * 50
                    head_id = f"{rounded_x}_{rounded_y}"
                    # Scale bubble by size
                    # Ensure min_area and max_area are defined for scaling
                    min_area, max_area = 1000, 30000
                    if is_near:
                        scale = min(1.5, max(1.0, (area - near_threshold) / (max_area - near_threshold)))
                    else:
                        scale = min(0.9, max(0.6, (area - min_detection_area) / (near_threshold - min_detection_area)))
                    # Dynamic Y-position (taller person = higher bubble)
                    bubble_y = y1 - int(height * 0.6)

                    # --- Updated current_axes calculation for overlap detection ---
                    # This needs to estimate the bubble size *before* drawing it,
                    # considering text wrapping.
                    estimated_font_scale = 0.6 * scale
                    estimated_font_thickness = max(1, int(2 * scale)) # Match bubble_drawer.py
                    # IMPORTANT: This must match max_bubble_content_width in bubble_drawer.py
                    estimated_max_content_width = int(220 * scale)

                    # Use a dummy long text to estimate the height after wrapping
                    # This is an approximation for overlap detection
                    dummy_long_text = "This is a very long dummy text to estimate the height of a wrapped bubble."
                    _, estimated_total_wrapped_text_height, _ = wrap_text(
                        dummy_long_text, cv2.FONT_HERSHEY_DUPLEX, estimated_font_scale, # Changed font here too
                        estimated_font_thickness, estimated_max_content_width
                    )

                    # Match padding values from bubble_drawer.py
                    dummy_padding_x, dummy_padding_y = int(35 * scale), int(25 * scale) # Adjusted padding
                    dummy_bubble_w = estimated_max_content_width + dummy_padding_x * 2
                    dummy_bubble_h = estimated_total_wrapped_text_height + dummy_padding_y * 2
                    current_axes = (dummy_bubble_w // 2, dummy_bubble_h // 2)
                    # --- End of updated current_axes calculation ---

                    # Prevent overlap
                    for _ in range(10):
                        bubble_top = bubble_y - current_axes[1] # Use current_axes for height
                        bubble_bottom = bubble_y + current_axes[1] # Use current_axes for height
                        overlap = False
                        for (tl, br, prev_axes) in used_bubble_boxes: # prev_axes is now returned
                            # Check for horizontal overlap and vertical overlap
                            # Use current_axes for the bubble being placed, and prev_axes for existing bubbles
                            if not (center_x + current_axes[0] < tl[0] or center_x - current_axes[0] > br[0]) and \
                               not (bubble_bottom < tl[1] or bubble_top > br[1]):
                                overlap = True
                                bubble_y -= int(20 * scale) # Move bubble up to avoid overlap
                                break
                        if not overlap:
                            break
                    if idx < max_bubbles:
                        if head_id not in head_thoughts:
                            head_thoughts[head_id] = random.choice(thought_texts)
                        thought = head_thoughts[head_id]
                        # Call the imported function
                        top_left, bottom_right, final_axes = draw_thought_bubble(frame, center_x, bubble_y, thought, scale)
                        used_bubble_boxes.append((top_left, bottom_right, final_axes)) # Store final_axes too
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
