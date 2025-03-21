import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Initialize global variables
points = []
zoom_scale = 1.0
display_image = None
original_image = None
filename = ""
distance_results = []  # To store distances and filenames
define_point_active = False  # Flag to track if "Define Point" is active
image_index = 0  # To track the image index in the folder

# Hardcoded path for the image folder
folder_path = "path/to/input/images"
output_file = "path/to/output/results.xlsx"

# Create a Matplotlib figure and axis
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title("Select two points")

def update_display():
    """Updates the image on the plot with any selected points and zooming."""
    global display_image, original_image, zoom_scale

    # If the image is grayscale, convert it to RGB (3 channels)
    if len(original_image.shape) == 2:  # Grayscale image (height, width)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

    # Convert to proper type if needed
    if original_image.dtype != np.uint8:
        original_image = np.uint8(original_image)

    # Resize the image based on the zoom scale
    display_image = cv2.resize(original_image, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_LINEAR)

    # Clear the current plot
    ax.clear()

    # Show the image with updated zoom scale
    ax.imshow(display_image)

    # Draw the selected points (scale them based on the zoom)
    for pt in points:
        x, y = pt
        scaled_x = int(x * zoom_scale)
        scaled_y = int(y * zoom_scale)
        # Draw the point on the resized image
        ax.plot(scaled_x, scaled_y, 'ro')  # 'ro' for red points

    # Redraw the figure to update the display
    plt.draw()

def select_points(event):
    """Store the points on click if Define Point is active."""
    global points
    if define_point_active and len(points) < 2:  # Only allow point selection if Define Point is active
        points.append((event.xdata, event.ydata))

    # Only update display once both points are defined
    if len(points) == 2:
        update_display()

def save_point(event):
    """Save the points and calculate distance."""
    global filename, points, distance_results

    if len(points) == 2:
        pt1, pt2 = points
        # Calculate distance in pixels
        distance = np.linalg.norm(np.array(pt1) - np.array(pt2))
        distance_results.append([filename, distance])
        
        # Clear points for the next image
        points.clear()

        # Update display after both points have been defined
        update_display()

        # Save the results to the Excel file
        save_to_excel()

        # Wait a short time before loading the next image
        plt.pause(0.5)

        # Move to the next image after processing
        load_next_image()

def save_to_excel():
    """Save results to Excel after each image is processed."""
    global distance_results
    if distance_results:
        df = pd.DataFrame(distance_results, columns=["filename", "distance_px_per_mm"])
        df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")

def load_next_image():
    """Load the next image for processing."""
    global image_index
    image_index += 1
    if image_index < len(image_files):
        process_image(image_files[image_index])
    else:
        # If all images have been processed, close the plot
        plt.close(fig)

def process_images(folder_path, output_file):
    global image_files
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'))])
    
    if image_files:
        process_image(image_files[0])

def process_image(image_filename):
    global original_image, filename, points, define_point_active

    filename = image_filename
    image_path = os.path.join(folder_path, filename)
    original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if original_image is None:
        print(f"Skipping {filename}, could not read the image.")
        return

    # Reset points and zoom scale for each image
    points.clear()
    zoom_scale = 1.0

    # Update the display with the current image
    update_display()

    # Enable Define Point
    define_point_active = False  # Disable until button is clicked

    # Create buttons
    ax_define = plt.axes([0.3, 0.01, 0.2, 0.05])
    btn_define = Button(ax_define, 'Define points')
    btn_define.on_clicked(lambda x: set_define_point_active(True))

    ax_save = plt.axes([0.55, 0.01, 0.2, 0.05])
    btn_save = Button(ax_save, 'Save points')
    btn_save.on_clicked(save_point)

    # Connect to the mouse event for selecting points
    fig.canvas.mpl_connect('button_press_event', select_points)

    # Show the image with Matplotlib
    plt.show()

def set_define_point_active(state):
    """Enable or disable the 'Define Point' state."""
    global define_point_active
    define_point_active = state

if __name__ == "__main__":
    process_images(folder_path, output_file)
