import os
import argparse
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# --- Type Hinting for Clarity ---
Point = Tuple[int, int]
AnalysisResult = Dict[str, Any]

class ObjectWidthAnalyzer:
    """
    An interactive GUI tool to measure the width of an object along its
    principal and perpendicular axes.
    """
    def __init__(self, input_dir: Path, output_dir: Path, params: Dict[str, Any]):
        self.input_dir = input_dir
        self.params = params
        
        # Create dedicated subdirectories for clean output
        self.img_output_dir = output_dir / "annotated_images"
        self.data_output_dir = output_dir / "data"
        self.img_output_dir.mkdir(parents=True, exist_ok=True)
        self.data_output_dir.mkdir(parents=True, exist_ok=True)
        
        supported = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        self.image_files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in supported])
        if not self.image_files:
            raise FileNotFoundError(f"No images found in {input_dir}")
        
        # --- State Management ---
        self.all_results: List[AnalysisResult] = []
        self.current_image_path: Optional[Path] = None
        self.original_image: Optional[np.ndarray] = None
        self.current_results: Dict[str, Any] = {} # Store results for the current image
        
        self.fig = None
        self.ax = None
        self.sliders: Dict[str, Slider] = {}

    def run(self):
        """Processes each image in the directory interactively."""
        print(f"Found {len(self.image_files)} images. Starting analysis...")
        for img_path in self.image_files:
            self.current_image_path = img_path
            self.original_image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if self.original_image is None:
                print(f"Warning: Could not read {img_path.name}. Skipping.")
                continue
            
            self._create_interactive_window()
        
        self._export_all_data()
        print("\nAll images processed. Results saved in:", self.data_output_dir)

    def _create_interactive_window(self):
        """Sets up the Matplotlib window with sliders and a button for one image."""
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        plt.subplots_adjust(bottom=0.3)
        
        self._create_widgets()
        self._update_plot() # Initial processing and display
        plt.show() # Blocks until the window is closed

    def _update_plot(self, val=None):
        """Gathers slider values, processes the image, and updates the plot."""
        current_params = {name: slider.val for name, slider in self.sliders.items()}
        
        # --- Run Full Analysis Pipeline ---
        processed_img, contours = self._segment_object(current_params)
        centroid = self._find_centroid(contours)
        
        # Find the primary (blue) and secondary (red) axes
        main_axis_pts, perp_axis_pts = self._find_axes(processed_img, centroid, current_params['angle'])
        
        # Measure widths along both axes
        main_axis_widths = self._measure_widths(self.ax, processed_img, main_axis_pts, 'blue')
        perp_axis_widths = self._measure_widths(self.ax, processed_img, perp_axis_pts, 'red')
        
        # --- Redraw the Plot ---
        self.ax.clear()
        self.ax.imshow(processed_img, cmap='gray')
        self.ax.set_title(f"Analyzing: {self.current_image_path.name}")
        self.ax.axis('off')
        
        # Draw main axes
        if main_axis_pts: self.ax.plot(*zip(*main_axis_pts), 'blue', linewidth=2)
        if perp_axis_pts: self.ax.plot(*zip(*perp_axis_pts), 'red', linewidth=2)
        
        # Store results for potential saving
        self.current_results = {
            "image_path": self.current_image_path,
            "main_axis_widths": main_axis_widths,
            "perp_axis_widths": perp_axis_widths,
            "fig": self.fig
        }
        
        self.fig.canvas.draw_idle()

    def _segment_object(self, p: Dict[str, Any]) -> Tuple[np.ndarray, List]:
        """Core image processing pipeline to create a binary mask of the object."""
        # Ensure odd kernel size for blur
        blur_ksize = max(3, int(p['blur']) | 1)
        kernel_size = max(1, int(p['kernel']))
        
        smoothed = cv2.GaussianBlur(self.original_image, (blur_ksize, blur_ksize), 0)
        grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        edges = np.uint8(np.clip(magnitude, 0, 255))
        
        _, thresholded = cv2.threshold(edges, int(p['canny_low']), int(p['canny_high']), cv2.THRESH_BINARY)
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        smoothed_edges = cv2.medianBlur(closed, 5)
        
        contours, _ = cv2.findContours(smoothed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filled_image = np.zeros_like(smoothed_edges)
        if contours:
            cv2.drawContours(filled_image, contours, -1, 255, thickness=cv2.FILLED)
            
        return filled_image, contours
    
    @staticmethod
    def _find_centroid(contours: List) -> Optional[Point]:
        """Finds the centroid of the largest contour."""
        if not contours: return None
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0: return None
        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    
    def _find_axes(self, image: np.ndarray, centroid: Point, angle_deg: float) -> Tuple[List[Point], List[Point]]:
        """Scans from the centroid to find the endpoints of the two main axes."""
        if centroid is None: return [], []
        
        # Primary axis (oriented by angle slider)
        angle_rad = np.deg2rad(angle_deg)
        start_pt = self._scan_in_direction(image, centroid, angle_rad)
        end_pt = self._scan_in_direction(image, centroid, angle_rad + np.pi) # Opposite direction
        main_axis_pts = [p for p in [start_pt, end_pt] if p is not None]
        
        # Perpendicular axis
        perp_start_pt = self._scan_in_direction(image, centroid, angle_rad + np.pi / 2)
        perp_end_pt = self._scan_in_direction(image, centroid, angle_rad - np.pi / 2)
        perp_axis_pts = [p for p in [perp_start_pt, perp_end_pt] if p is not None]

        return main_axis_pts, perp_axis_pts

    @staticmethod
    def _scan_in_direction(image: np.ndarray, start_pt: Point, angle_rad: float) -> Optional[Point]:
        """Finds the first non-object pixel along a ray from a starting point."""
        h, w = image.shape
        max_dist = int(math.hypot(w, h))
        for r in range(1, max_dist):
            x = int(start_pt[0] + r * math.cos(angle_rad))
            y = int(start_pt[1] + r * math.sin(angle_rad))
            
            if not (0 <= y < h and 0 <= x < w): break # Out of bounds
            if image[y, x] == 0: # Found the edge (black pixel)
                return x, y
        return None

    def _measure_widths(self, ax: Axes, image: np.ndarray, axis_pts: List[Point], color: str) -> List[float]:
        """Measures the object width at intervals along a given axis."""
        if len(axis_pts) < 2: return []

        widths = []
        p1, p2 = axis_pts
        axis_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if axis_len == 0: return []
        
        num_measurements = int(axis_len / self.params['gap'])
        if num_measurements < 1: return []
        
        # Unit vector along the axis
        vx, vy = (p2[0] - p1[0]) / axis_len, (p2[1] - p1[1]) / axis_len
        
        for i in range(1, num_measurements):
            # Point along the axis to measure from
            t = i / num_measurements
            mx, my = p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])
            
            # Perpendicular vector
            px, py = -vy, vx
            
            # Scan in both perpendicular directions
            p_start = self._scan_in_direction(image, (mx, my), math.atan2(py, px))
            p_end = self._scan_in_direction(image, (mx, my), math.atan2(-py, -px))
            
            if p_start and p_end:
                width = math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])
                widths.append(width)
                ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], color, linewidth=0.5)

        return widths
    
    def _save_and_close(self, event):
        """Saves results and closes the current window."""
        if not self.current_results:
            print("No results to save.")
            plt.close(self.fig)
            return

        # Append data to the master list
        self.all_results.append({
            "filename": self.current_image_path.name,
            "main_axis_widths_px": self.current_results['main_axis_widths'],
            "perp_axis_widths_px": self.current_results['perp_axis_widths']
        })
        
        # Save the annotated figure
        save_path_img = self.img_output_dir / f"{self.current_image_path.stem}_analyzed.png"
        self.current_results['fig'].savefig(save_path_img, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved annotated image to {save_path_img}")
        
        plt.close(self.fig)

    def _export_all_data(self):
        """Exports all collected data to a single Excel file."""
        if not self.all_results:
            print("No data collected to export.")
            return

        # Unpack the lists of widths into a long format DataFrame
        long_format_data = []
        for res in self.all_results:
            for width in res['main_axis_widths_px']:
                long_format_data.append([res['filename'], 'main_axis', width])
            for width in res['perp_axis_widths_px']:
                long_format_data.append([res['filename'], 'perp_axis', width])
        
        df = pd.DataFrame(long_format_data, columns=['filename', 'axis', 'width_px'])
        
        save_path_excel = self.data_output_dir / "all_width_measurements.xlsx"
        df.to_excel(save_path_excel, index=False)
        print(f"Exported all data to {save_path_excel}")

    def _create_widgets(self):
        """Creates and lays out all the sliders and the button."""
        p = self.params
        widget_color = 'lightgoldenrodyellow'
        
        ax_angle = plt.axes([0.25, 0.20, 0.65, 0.022], facecolor=widget_color)
        ax_kernel = plt.axes([0.25, 0.17, 0.65, 0.022], facecolor=widget_color)
        ax_cl = plt.axes([0.25, 0.14, 0.65, 0.022], facecolor=widget_color)
        ax_ch = plt.axes([0.25, 0.11, 0.65, 0.022], facecolor=widget_color)
        ax_blur = plt.axes([0.25, 0.08, 0.65, 0.022], facecolor=widget_color)
        ax_btn = plt.axes([0.8, 0.9, 0.1, 0.04])

        self.sliders['angle'] = Slider(ax_angle, 'Angle', -90, 90, valinit=p['angle'], valstep=0.5)
        self.sliders['kernel'] = Slider(ax_kernel, 'Kernel Size', 1, 20, valinit=p['kernel'], valstep=1)
        self.sliders['canny_low'] = Slider(ax_cl, 'Thresh Low', 0, 100, valinit=p['canny_low'], valstep=1)
        self.sliders['canny_high'] = Slider(ax_ch, 'Thresh High', 0, 100, valinit=p['canny_high'], valstep=1)
        self.sliders['blur'] = Slider(ax_blur, 'Blur Kernel', 1, 21, valinit=p['blur'], valstep=2)
        self.button = Button(ax_btn, 'Save & Next')

        for slider in self.sliders.values():
            slider.on_changed(self._update_plot)
        self.button.on_clicked(self._save_and_close)

def main():
    parser = argparse.ArgumentParser(description="Interactively analyze object width.")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input directory.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory.")
    
    # Add arguments for all initial parameters
    parser.add_argument("--canny-low", type=int, default=0)
    parser.add_argument("--canny-high", type=int, default=8)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--blur", type=int, default=5)
    parser.add_argument("--angle", type=float, default=0)
    parser.add_argument("--gap", type=int, default=50, help="Gap between width measurements in pixels.")
    args = parser.parse_args()

    params = {k: v for k, v in vars(args).items() if k not in ['input', 'output']}
    
    try:
        analyzer = ObjectWidthAnalyzer(args.input, args.output, params)
        analyzer.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
