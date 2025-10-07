import os
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import Button

# Type hinting for clarity
Point = Tuple[float, float]

class PixelCalibrator:
    """
    A GUI tool to measure a known distance in pixels across a series of images.

    This tool loads images from a specified directory, allows a user to select
    two points, calculates the pixel distance between them, and saves the results
    to an Excel file. This is useful for creating a scale reference (e.g., pixels per mm)
    for further image analysis.
    """
    def __init__(self, image_dir: Path, output_dir: Path, known_distance_mm: int):
        """
        Initializes the PixelCalibrator application.

        Args:
            image_dir (Path): Path to the directory containing images.
            output_dir (Path): Path to the directory where results will be saved.
            known_distance_mm (int): The known real-world distance (in mm) being measured.
                                     This is used for naming the output column.
        """
        if not image_dir.is_dir():
            raise FileNotFoundError(f"The specified image directory does not exist: {image_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir = image_dir
        self.known_distance_mm = known_distance_mm

        # Dynamically create the output filename
        today_str = datetime.now().strftime('%Y-%m-%d')
        self.output_file = output_dir / f"{today_str}_pixel_distances.xlsx"

        # Discover and sort image files
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        self.image_files: List[Path] = sorted(
            [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in supported_formats]
        )
        if not self.image_files:
            raise FileNotFoundError(f"No images found in the directory: {image_dir}")

        # --- State variables ---
        self.current_image_index: int = 0
        self.points: List[Point] = []
        self.distance_results: List[List] = []
        self.define_point_active: bool = False

        # --- Image data ---
        self.original_image: Optional[np.ndarray] = None

        # --- Matplotlib UI elements ---
        self.fig: Figure
        self.ax: Axes
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def run(self):
        """Starts the image processing workflow."""
        self._setup_ui()
        self._load_next_image()
        plt.show()

    def _setup_ui(self):
        """Sets up the Matplotlib UI buttons."""
        plt.subplots_adjust(bottom=0.15)
        self.ax.set_title("Click 'Define Points', then select two points on the image.")

        ax_define = self.fig.add_axes([0.3, 0.05, 0.2, 0.05])
        btn_define = Button(ax_define, 'Define Points')
        btn_define.on_clicked(self._activate_define_mode)

        ax_save = self.fig.add_axes([0.55, 0.05, 0.2, 0.05])
        btn_save = Button(ax_save, 'Save & Next Image')
        btn_save.on_clicked(self._save_and_next)

    def _load_next_image(self):
        """Loads and displays the next image in the sequence."""
        if self.current_image_index >= len(self.image_files):
            print("All images processed. Closing application.")
            plt.close(self.fig)
            return

        image_path = self.image_files[self.current_image_index]
        self.original_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if self.original_image is None:
            print(f"Warning: Could not read image {image_path.name}. Skipping.")
            self.current_image_index += 1
            self._load_next_image()
            return

        # Ensure image is in a displayable format (RGB, uint8)
        if len(self.original_image.shape) == 2:
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2RGB)
        if self.original_image.dtype != np.uint8:
            self.original_image = self.original_image.astype(np.uint8)

        # Reset state for the new image
        self.points = []
        self.define_point_active = False
        self._update_display()

    def _update_display(self):
        """Redraws the image and any selected points on the canvas."""
        self.ax.clear()
        self.ax.imshow(self.original_image)
        current_filename = self.image_files[self.current_image_index].name
        self.ax.set_title(f"Processing: {current_filename}")

        # Draw the selected points and line
        for x, y in self.points:
            self.ax.plot(x, y, 'ro', markersize=5)

        if len(self.points) == 2:
            pt1, pt2 = self.points
            self.ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-')

        self.fig.canvas.draw_idle()

    def _on_click(self, event: MouseEvent):
        """Handles mouse click events on the image canvas."""
        if not self.define_point_active or event.inaxes != self.ax or len(self.points) >= 2:
            return

        self.points.append((event.xdata, event.ydata))
        print(f"Point {len(self.points)} selected at: ({event.xdata:.2f}, {event.ydata:.2f})")
        self._update_display()

    def _activate_define_mode(self, event: MouseEvent):
        """Activates point definition mode and clears any existing points."""
        print("Define mode activated. Please select two points.")
        self.define_point_active = True
        self.points = []
        self._update_display()

    def _save_and_next(self, event: MouseEvent):
        """Saves the calculated distance and loads the next image."""
        if len(self.points) != 2:
            print("Error: Please select exactly two points before saving.")
            return

        pt1, pt2 = self.points
        distance = np.linalg.norm(np.array(pt1) - np.array(pt2))

        filename = self.image_files[self.current_image_index].name
        self.distance_results.append([filename, distance])
        print(f"Saved distance for {filename}: {distance:.2f} pixels.")

        self._save_to_excel()
        self.current_image_index += 1
        self._load_next_image()

    def _save_to_excel(self):
        """Saves the collected distance results to an Excel file."""
        if not self.distance_results:
            return

        try:
            # Create a dynamic column name based on the known distance
            column_name = f"distance_{self.known_distance_mm}_mm_px"
            df = pd.DataFrame(self.distance_results, columns=["filename", column_name])
            df.to_excel(self.output_file, index=False)
            print(f"Results successfully updated in {self.output_file}")
        except Exception as e:
            print(f"Error saving to Excel file: {e}")

def main():
    """Main function to parse arguments and run the processor."""
    parser = argparse.ArgumentParser(
        description="A GUI tool to measure a known pixel distance in a series of images for calibration."
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to the directory containing images."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Path to the directory where the output Excel file will be saved."
    )
    parser.add_argument(
        "-d", "--distance",
        type=int,
        default=30,
        help="The known real-world distance (in mm) being measured. Default: 30."
    )
    args = parser.parse_args()

    try:
        calibrator = PixelCalibrator(
            image_dir=args.input,
            output_dir=args.output,
            known_distance_mm=args.distance
        )
        calibrator.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
