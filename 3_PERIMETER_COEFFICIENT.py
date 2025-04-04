import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
import math
import openpyxl

# Define the directory paths
directory_path = 
output_dir = 
output_dir_excel = 
os.makedirs(output_dir, exist_ok=True)

# Get a list of image files
image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp'))]

if not image_files:
    print("No images found in the directory.")
    exit()

canny_lower_threshold = 
canny_upper_threshold = 
canny_lower_boundary =
canny_upper_boundary = 
kernel_size = 
kernel_lower_boundary = 
kernel_upper_boundary = 
kernel_step =
blur_threshold =
blur_step =
blur_lower_boundary =
blur_upper_boundary = 
perpendicular_gap = 
directional_gap = 

def process_image(image, lower_threshold, upper_threshold, kernel_size, blur_ksize = 5):
    """Apply Sobel edge detection, morphological closing, contour filling, and edge smoothing."""

    # Apply Gaussian blur to smooth edges before Sobel
    smoothed = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    
    # Apply Sobel edge detection
    grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the magnitude of the gradient
    magnitude = cv2.magnitude(grad_x, grad_y)
    
    # Normalize the magnitude to the range [0, 255] for visualization
    edges = np.uint8(np.clip(magnitude, 0, 255))
    
    # Apply a threshold to the edges (optional, based on lower and upper threshold)
    _, edges = cv2.threshold(edges, int(lower_threshold), int(upper_threshold), cv2.THRESH_BINARY)

    # Create a kernel for morphological closing
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply morphological closing to fill small gaps in edges
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Apply median blur to further smoothen edges
    smoothed_edges = cv2.medianBlur(closed_edges, 5)

    # Fill the detected objects
    filled_image = np.zeros_like(smoothed_edges)
    contours, _ = cv2.findContours(smoothed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cv2.drawContours(filled_image, [contour], -1, (255), thickness=cv2.FILLED)

    return filled_image, contours

def find_centroid(contours):
    """Find and return the centroid of the largest contour."""
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    
    if M["m00"] == 0:
        return None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    return (cx, cy)

def save_filled_object(fig, image_name):
    """Save the displayed figure as a high-resolution image."""
    base_name = os.path.splitext(image_name)[0]
    save_path = os.path.join(output_dir, f"{base_name}_filled.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"Saved: {save_path}")

def calculate_length(point1, point2):
    """Calculate the Euclidean distance between two points."""
    if point1 is None or point2 is None:
        return 0
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def scan_from_centroid(filled_object, centroid, num_closest_points = 1, angle_value = 0):
    """Scan from centroid and find the 100 closest edge points, then return their average."""
    if centroid is None:
        return None

    cx, cy = centroid
    max_radius = min(filled_object.shape[0], filled_object.shape[1]) // 2
    edge_points = []

    for radius in range(1, max_radius):
        # Calculate the lower and upper bounds of the angle range (in radians)
        lower_angle = np.deg2rad(angle_value - 5)  # 35 degrees if angle_value is 45 degrees
        upper_angle = np.deg2rad(angle_value + 5)  # 55 degrees if angle_value is 45 degrees
        
        # Generate evenly spaced angles between lower_angle and upper_angle
        for angle in np.linspace(lower_angle, upper_angle, 100):
            x = int(cx + radius * np.cos(angle))
            y = int(cy + radius * np.sin(angle))
            
            if 0 <= x < filled_object.shape[1] and 0 <= y < filled_object.shape[0]:
                if filled_object[y, x] == 0:
                    distance = calculate_length(centroid, (x, y))
                    edge_points.append((x, y, distance))

    # Sort edge points by distance to centroid
    edge_points.sort(key=lambda point: point[2])

    # Select the closest 'num_closest_points' points
    closest_points = edge_points[:num_closest_points]

    # Compute the average of the closest points
    x = sum(point[0] for point in closest_points) / len(closest_points)
    y = sum(point[1] for point in closest_points) / len(closest_points)

    return (int(x), int(y))  # Return the average as a tuple

def scan_opposite_from_centroid(filled_object, centroid, edge_point):
    """Scan from centroid opposite to the edge point"""
    if centroid is None or edge_point is None:
        return None

    cx, cy = centroid
    ex, ey = edge_point

    # Directional vector to the edge point
    direction_x = ex - cx
    direction_y = ey - cy

    # Opposite direction
    opposite_x = cx - direction_x
    opposite_y = cy - direction_y

    max_radius = min(filled_object.shape[0], filled_object.shape[1]) // 2
    for radius in range(1, max_radius):
        for angle in np.linspace(0, 2 * np.pi, 360):
            x = int(opposite_x + radius * np.cos(angle))
            y = int(opposite_y + radius * np.sin(angle))

            if 0 <= x < filled_object.shape[1] and 0 <= y < filled_object.shape[0]:
                if filled_object[y, x] == 0:
                    return (x, y)

    return None

def scan_perpendicular_from_centroid(filled_object, centroid, edge_point, gap = perpendicular_gap):
    """Scan from the centroid in two perpendicular directions (clockwise and counterclockwise)
       with a 50-pixel gap in both directions."""
    if centroid is None or edge_point is None:
        return None, None

    cx, cy = centroid
    ex, ey = edge_point

    # Calculate the direction from centroid to edge point
    direction_x = ex - cx
    direction_y = ey - cy

    # Calculate perpendicular directions (clockwise and counterclockwise)
    # Clockwise perpendicular vector
    perp_cw_x = -direction_y
    perp_cw_y = direction_x
    
    # Counterclockwise perpendicular vector
    perp_ccw_x = direction_y
    perp_ccw_y = -direction_x
    
    # Normalize both perpendicular vectors
    norm_cw = math.sqrt(perp_cw_x**2 + perp_cw_y**2)
    norm_ccw = math.sqrt(perp_ccw_x**2 + perp_ccw_y**2)
    
    # Scale the vectors to a reasonable step size for scanning
    perp_cw_x /= norm_cw
    perp_cw_y /= norm_cw
    perp_ccw_x /= norm_ccw
    perp_ccw_y /= norm_ccw
    
    max_radius = min(filled_object.shape[0], filled_object.shape[1]) // 2

    # Scan along the perpendicular direction clockwise with a 50-pixel gap
    x_cw_start = int(cx + gap * perp_cw_x)
    y_cw_start = int(cy + gap * perp_cw_y)

    for radius in range(1, max_radius):
        x_cw = int(x_cw_start + radius * perp_cw_x)
        y_cw = int(y_cw_start + radius * perp_cw_y)

        if 0 <= x_cw < filled_object.shape[1] and 0 <= y_cw < filled_object.shape[0]:
            if filled_object[y_cw, x_cw] == 0:  # Black pixel detected
                break
        else:
            x_cw, y_cw = None, None

    # Scan along the perpendicular direction counterclockwise with a 50-pixel gap
    x_ccw_start = int(cx + gap * perp_ccw_x)
    y_ccw_start = int(cy + gap * perp_ccw_y)

    for radius in range(1, max_radius):
        x_ccw = int(x_ccw_start + radius * perp_ccw_x)
        y_ccw = int(y_ccw_start + radius * perp_ccw_y)

        if 0 <= x_ccw < filled_object.shape[1] and 0 <= y_ccw < filled_object.shape[0]:
            if filled_object[y_ccw, x_ccw] == 0:  # Black pixel detected
                break
        else:
            x_ccw, y_ccw = None, None

    return (x_cw, y_cw), (x_ccw, y_ccw)

def get_line_points(centroid, edge_point):
    """Generate all points along the line from the centroid to an edge point."""
    cx, cy = centroid
    ex, ey = edge_point
    points = []

    # Calculate the direction from centroid to edge
    dx = ex - cx
    dy = ey - cy
    steps = max(abs(dx), abs(dy))

    # Get the unit step size in both x and y directions
    step_x = dx / steps
    step_y = dy / steps

    # Trace the line and collect the points
    for i in range(steps + 1):
        x = int(cx + i * step_x)
        y = int(cy + i * step_y)
        points.append((x, y))

    return points

# Function to plot perpendiculars and save their lengths
def plot_perpendiculars(ax, red_line_coords, filled_object, gap = perpendicular_gap):
    """Plot perpendiculars along the red line, saving each perpendicular's length to a list."""
    perpendicular_lengths = []

    for i in range(0, len(red_line_coords), gap):
        segment = red_line_coords[i - directional_gap : i + directional_gap]
        if len(segment) < 2:
            continue

        # Fit a local direction using the first and last point of the segment
        x1, y1 = segment[0]
        x2, y2 = segment[-1]

        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx**2 + dy**2)

        if length == 0:
            continue  # Skip if no movement

        # Normalize the direction vector
        dx /= length
        dy /= length

        # Compute perpendicular direction (90-degree rotation)
        perp_x = -dy
        perp_y = dx

        # Midpoint where perpendicular should be drawn
        mx, my = red_line_coords[i]

        # Scan outward until a black pixel is found (for both directions)
        def find_black_pixel(x_start, y_start, step_x, step_y):
            """Move in a given direction until a black pixel is found."""
            for step in range(1, min(filled_object.shape) // 2):
                x = int(x_start + step * step_x)
                y = int(y_start + step * step_y)

                if 0 <= x < filled_object.shape[1] and 0 <= y < filled_object.shape[0]:
                    if filled_object[y, x] == 0:  # Black pixel found
                        return x, y
            return None

        # Find black pixels in both perpendicular directions
        perp_start = find_black_pixel(mx, my, perp_x, perp_y)
        perp_end = find_black_pixel(mx, my, -perp_x, -perp_y)

        # Calculate the length of the perpendicular if both ends are found
        if perp_start and perp_end:
            length_perpendicular = calculate_length(perp_start, perp_end)
            perpendicular_lengths.append(length_perpendicular)

            # Draw the perpendicular line
            ax.plot([perp_start[0], perp_end[0]], [perp_start[1], perp_end[1]], 'blue', linewidth=1)

    return perpendicular_lengths  # Return the list of lengths

def plot_blue_perpendiculars(ax, blue_line_coords, filled_object, gap=perpendicular_gap):
    """Plot perpendiculars along the blue line, saving each perpendicular's length to a list."""
    blue_perpendicular_lengths = []

    for i in range(0, len(blue_line_coords), gap):
        segment = blue_line_coords[i - directional_gap : i + directional_gap]
        if len(segment) < 2:
            continue

        # Fit a local direction using the first and last point of the segment
        x1, y1 = segment[0]
        x2, y2 = segment[-1]

        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx**2 + dy**2)

        if length == 0:
            continue  # Skip if no movement

        # Normalize the direction vector
        dx /= length
        dy /= length

        # Compute perpendicular direction (90-degree rotation)
        perp_x = -dy
        perp_y = dx

        # Midpoint where perpendicular should be drawn
        mx, my = blue_line_coords[i]

        # Scan outward until a black pixel is found (for both directions)
        def find_black_pixel(x_start, y_start, step_x, step_y):
            """Move in a given direction until a black pixel is found."""
            for step in range(1, min(filled_object.shape) // 2):  # Max radius limit
                x = int(x_start + step * step_x)
                y = int(y_start + step * step_y)

                if 0 <= x < filled_object.shape[1] and 0 <= y < filled_object.shape[0]:
                    if filled_object[y, x] == 0:  # Black pixel found
                        return x, y
            return None  # No black pixel found

        # Find black pixels in both perpendicular directions
        perp_start = find_black_pixel(mx, my, perp_x, perp_y)
        perp_end = find_black_pixel(mx, my, -perp_x, -perp_y)

        # Calculate the length of the perpendicular if both ends are found
        if perp_start and perp_end:
            length_perpendicular = calculate_length(perp_start, perp_end)
            blue_perpendicular_lengths.append(length_perpendicular)  # Append each length

            # Draw the perpendicular line
            ax.plot([perp_start[0], perp_end[0]], [perp_start[1], perp_end[1]], 'red', linewidth=1)

    return blue_perpendicular_lengths  # Return the list of lengths

# Initialize Excel workbook and sheet
def save_perpendicular_lengths_to_excel(image_file, blue_lengths, red_lengths):
    """Save individual perpendicular lengths for each image to an Excel file."""
    # Initialize workbook for each image
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["filename", "y_axis_length_px", "x_axis_length_px"])

    # Find the maximum length between blue and red perpendiculars to align them properly
    max_len = max(len(blue_lengths), len(red_lengths))

    # Loop through the lengths and save each one to the appropriate column
    for i in range(max_len):
        blue_length = blue_lengths[i] if i < len(blue_lengths) else None
        red_length = red_lengths[i] if i < len(red_lengths) else None
        ws.append([image_file, blue_length, red_length])

    # Save the Excel file with the image's filename
    excel_path = os.path.join(output_dir_excel, f"{os.path.splitext(image_file)[0]}.xlsx")
    wb.save(excel_path)
    print(f"Excel file saved: {excel_path}")

# Update function for sliders
def update(val, im_obj, ax, fig):
    # Get the current slider values
    kernel_size = int(slider_kernel.val)
    canny_lower = int(slider_canny_low.val)
    canny_upper = int(slider_canny_high.val)
    blur_ksize = int(slider_blur.val)
    angle_value = int(slider_angle.val)

    # Process the image with the current slider values
    processed_img, contours = process_image(image, canny_lower, canny_upper, kernel_size, blur_ksize)
    
    # Find the centroid of the contours
    centroid = find_centroid(contours)
    
    # Initialize coordinates lists
    blue_line_coords = []
    red_line_coords = []

    # Scan from centroid and find the first edge point (blue line)
    edge_point = scan_from_centroid(processed_img, centroid, num_closest_points=1, angle_value=angle_value)
    if edge_point:
        # Get all points from centroid to edge point
        blue_line_coords.extend(get_line_points(centroid, edge_point))

        # Scan for the opposite edge (blue line)
        opposite_edge = scan_opposite_from_centroid(processed_img, centroid, edge_point)
        if opposite_edge:
            # Get all points from centroid to opposite edge
            blue_line_coords.extend(get_line_points(centroid, opposite_edge))

        # Scan for perpendicular edges (red line)
        cw_edge, ccw_edge = scan_perpendicular_from_centroid(processed_img, centroid, edge_point)
        if cw_edge:
            # Get all points from centroid to clockwise edge
            red_line_coords.extend(get_line_points(centroid, cw_edge))
        if ccw_edge:
            # Get all points from centroid to counterclockwise edge
            red_line_coords.extend(get_line_points(centroid, ccw_edge))

    # Update image data
    im_obj.set_data(processed_img)
    ax.clear()  # Clear the axis

    # Recreate the image and draw lines
    im_obj = ax.imshow(processed_img, cmap='gray')
    ax.axis('off')

    # Draw blue and red lines
    for i in range(len(blue_line_coords) - 1):
        ax.plot([blue_line_coords[i][0], blue_line_coords[i + 1][0]],
                [blue_line_coords[i][1], blue_line_coords[i + 1][1]], 'blue', linewidth=2)

    for i in range(len(red_line_coords) - 1):
        ax.plot([red_line_coords[i][0], red_line_coords[i + 1][0]],
                [red_line_coords[i][1], red_line_coords[i + 1][1]], 'red', linewidth=2)

    # Plot blue and red perpendiculars
    blue_perpendicular_lengths = plot_blue_perpendiculars(ax, blue_line_coords, processed_img, gap=perpendicular_gap)
    perpendicular_lengths = plot_perpendiculars(ax, red_line_coords, processed_img, gap=perpendicular_gap)

    # Save data to Excel
    save_perpendicular_lengths_to_excel(image_file, blue_perpendicular_lengths, perpendicular_lengths)

    # Save the figure
    save_filled_object(fig, image_file)

    # Redraw the canvas to reflect changes
    fig.canvas.draw()

# Main loop through images
for image_file in image_files:
    image_path = os.path.join(directory_path, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Failed to load image: {image_path}")
        continue

    # Process image
    filled_object, contours = process_image(image, canny_lower_threshold, canny_upper_threshold, kernel_size)

    # Find centroid
    centroid = find_centroid(contours)

    # Initialize coordinates lists
    blue_line_coords = []
    red_line_coords = []

    # Scan from centroid and find the first edge point (blue line)
    edge_point = scan_from_centroid(filled_object, centroid)
    if edge_point:
        # Get all points from centroid to edge point
        blue_line_coords.extend(get_line_points(centroid, edge_point))

        # Scan for the opposite edge (blue line)
        opposite_edge = scan_opposite_from_centroid(filled_object, centroid, edge_point)
        if opposite_edge:
            # Get all points from centroid to opposite edge
            blue_line_coords.extend(get_line_points(centroid, opposite_edge))

        # Scan for perpendicular edges (red line)
        cw_edge, ccw_edge = scan_perpendicular_from_centroid(filled_object, centroid, edge_point)
        if cw_edge:
            # Get all points from centroid to clockwise edge
            red_line_coords.extend(get_line_points(centroid, cw_edge))
        if ccw_edge:
            # Get all points from centroid to counterclockwise edge
            red_line_coords.extend(get_line_points(centroid, ccw_edge))

    # Create the figure
    fig = plt.figure(figsize=(16, 10))

    # Create an axis for the image
    ax = fig.add_subplot(111)
    im_obj = ax.imshow(filled_object, cmap='gray')
    ax.axis('off')

    # Draw lines
    for i in range(len(blue_line_coords) - 1):
        ax.plot([blue_line_coords[i][0], blue_line_coords[i + 1][0]],
                [blue_line_coords[i][1], blue_line_coords[i + 1][1]], 'blue', linewidth=2)

    for i in range(len(red_line_coords) - 1):
        ax.plot([red_line_coords[i][0], red_line_coords[i + 1][0]],
                [red_line_coords[i][1], red_line_coords[i + 1][1]], 'red', linewidth=2)

    # Plot blue perpendiculars and get all their lengths
    blue_perpendicular_lengths = plot_blue_perpendiculars(ax, blue_line_coords, filled_object, gap=perpendicular_gap)

    # Plot red perpendiculars and get all their lengths
    perpendicular_lengths = plot_perpendiculars(ax, red_line_coords, filled_object, gap=perpendicular_gap)

    # Save each perpendicular length to Excel for both blue and red lines
    save_perpendicular_lengths_to_excel(image_file, blue_perpendicular_lengths, perpendicular_lengths)

    # Save the figure (optional)
    save_filled_object(fig, image_file)

    # Define slider positions
    ax_kernel = plt.axes([0.25, 0.13, 0.65, 0.03])
    ax_canny_low = plt.axes([0.25, 0.09, 0.65, 0.03])
    ax_canny_high = plt.axes([0.25, 0.05, 0.65, 0.03])
    ax_blur = plt.axes([0.25, 0.01, 0.65, 0.03])
    ax_angle = plt.axes([0.25, 0.17, 0.65, 0.03])

    # Create sliders
    slider_kernel = Slider(ax_kernel, 'Kernel', kernel_lower_boundary, kernel_upper_boundary, valinit=kernel_size, valstep = kernel_step)
    slider_canny_low = Slider(ax_canny_low, 'Canny Low', canny_lower_boundary, canny_upper_boundary, valinit = canny_lower_threshold)
    slider_canny_high = Slider(ax_canny_high, 'Canny High', canny_lower_boundary, canny_upper_boundary, valinit = canny_upper_threshold)
    slider_blur = Slider(ax_blur, 'Blur', blur_lower_boundary, blur_upper_boundary, valinit = blur_threshold, valstep = 2)
    slider_angle = Slider(ax_angle, 'Angle', -45, 45, valinit = 0, valstep = 0.5)

    # Attach update function to sliders with im_obj, ax, and fig as arguments
    slider_kernel.on_changed(lambda val: update(val, im_obj, ax, fig))
    slider_canny_low.on_changed(lambda val: update(val, im_obj, ax, fig))
    slider_canny_high.on_changed(lambda val: update(val, im_obj, ax, fig))
    slider_blur.on_changed(lambda val: update(val, im_obj, ax, fig))
    slider_angle.on_changed(lambda val: update(val, im_obj, ax, fig))

    # Show the plot
    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9)
    plt.show()
    