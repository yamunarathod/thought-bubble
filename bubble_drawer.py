import cv2
import math

# --- Bubble Design Parameters (can be easily modified here) ---
# Colors (BGR format)
# Equivalent to Tailwind's bg-yellow-100 (rgb(255, 255, 179))
YELLOW_100 = (179, 255, 255)
# Equivalent to Tailwind's border-orange-400 (rgb(60, 146, 251))
ORANGE_400 = (60, 146, 251)
# Adjusted text color to a dark gray (less dark than pure black)
TEXT_COLOR = (50, 50, 50) # Dark gray

# --- Helper function for text wrapping ---
def wrap_text(text, font, font_scale, font_thickness, max_width):
    """
    Wraps text to fit within a specified maximum width.
    Returns a list of lines, total height of wrapped text, and height of a single line.
    """
    words = text.split(' ')
    lines = []
    current_line = []
    current_line_width = 0
    # Estimate line height using a common character, increased for more line spacing
    line_height = cv2.getTextSize("Tg", font, font_scale, font_thickness)[0][1] + int(10 * font_scale) # Increased buffer

    for word in words:
        # Check width of current line + new word
        test_line = " ".join(current_line + [word])
        test_line_width, _ = cv2.getTextSize(test_line, font, font_scale, font_thickness)[0]

        if test_line_width < max_width:
            current_line.append(word)
            current_line_width = test_line_width
        else:
            # If the word itself is wider than max_width, put it on its own line
            if not current_line: # If current_line is empty, this word is too long for a line
                lines.append(word) # Add the long word as a single line
                current_line = []
                current_line_width = 0
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_line_width, _ = cv2.getTextSize(word, font, font_scale, font_thickness)[0]

    if current_line: # Add the last line if it's not empty
        lines.append(" ".join(current_line))

    total_height = len(lines) * line_height
    return lines, total_height, line_height

# --- Bubble Drawing Function ---
def draw_thought_bubble(img, x, y, text, scale=1.0):
    """
    Draws a solid oval-shaped thought bubble on the image, with text wrapping.
    Returns the top-left, bottom-right corners, and axes of the bubble for overlap detection.
    """
    font = cv2.FONT_HERSHEY_DUPLEX # Changed font family
    font_scale = 0.6 * scale
    # Adjusted font thickness for semibold effect
    font_thickness = max(1, int(2 * scale)) # Adjusted for semibold

    # Define maximum content width for the bubble (e.g., 220 pixels at scale 1.0)
    max_bubble_content_width = int(220 * scale)

    # Wrap the text
    wrapped_lines, total_wrapped_text_height, line_height = wrap_text(
        text, font, font_scale, font_thickness, max_bubble_content_width
    )

    # Calculate bubble dimensions based on wrapped text
    # Increased padding for better spacing all around
    padding_x, padding_y = int(35 * scale), int(25 * scale) # Increased padding
    bubble_w = max_bubble_content_width + padding_x * 2
    bubble_h = total_wrapped_text_height + padding_y * 2

    center = (x, y - bubble_h // 2)
    axes = (bubble_w // 2, bubble_h // 2)

    # Draw yellow-100 oval bubble fill
    cv2.ellipse(img, center, axes, 0, 0, 360, YELLOW_100, -1)
    # Draw solid border for the main oval
    cv2.ellipse(img, center, axes, 0, 0, 360, ORANGE_400, 2) # Thickness 2

    # Draw tail circles
    tail1_center = (x - int(15 * scale), y + int(10 * scale))
    tail1_radius = int(8 * scale)
    tail2_center = (x - int(5 * scale), y + int(18 * scale))
    tail2_radius = int(5 * scale)

    # Draw fill for tail circles
    cv2.circle(img, tail1_center, tail1_radius, YELLOW_100, -1)
    cv2.circle(img, tail2_center, tail2_radius, YELLOW_100, -1)

    # Draw solid border for tail circles
    cv2.circle(img, tail1_center, tail1_radius, ORANGE_400, 1) # Thickness 1
    cv2.circle(img, tail2_center, tail2_radius, ORANGE_400, 1) # Thickness 1

    # Draw text line by line
    # Calculate the starting Y position for the first line of text
    # It should be centered vertically within the bubble's content area
    text_start_y = center[1] - total_wrapped_text_height // 2 + line_height // 2 # Adjust for baseline

    for i, line in enumerate(wrapped_lines):
        # Get the width of the current line to center it
        line_w, _ = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
        line_x = center[0] - line_w // 2 # Center each line horizontally
        line_y = text_start_y + i * line_height # Position for current line

        cv2.putText(img, line, (line_x, line_y), font, font_scale, TEXT_COLOR, font_thickness) # Use TEXT_COLOR

    # Return top-left, bottom-right corners, and axes for overlap detection
    top_left = (center[0] - axes[0], center[1] - axes[1])
    bottom_right = (center[0] + axes[0], center[1] + axes[1])
    return top_left, bottom_right, axes
