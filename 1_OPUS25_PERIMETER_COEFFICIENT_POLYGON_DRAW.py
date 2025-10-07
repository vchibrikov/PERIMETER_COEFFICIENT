import os
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent, KeyEvent, ScrollEvent
from PIL import Image

# Type alias for a point for better readability
Point = Tuple[float, float]

class RegionSelector:
    """
    A helper class to manage interactive polygon drawing on a Matplotlib axis.
    Handles mouse clicks for drawing and scrolling for zooming.
    """
    def __init__(self, ax: Axes):
        self.ax = ax
        self.polygon_points: List[Point] = []
        self.line = Line2D([], [], color='cyan', linewidth=2, marker='o', markersize=5)
        self.ax.add_line(self.line)
        self.is_closed = False

        # Connect canvas events to methods
        self.cid_click = self.ax.figure.canvas.mpl_connect('button_press_event', self._on_click)
        self.cid_scroll = self.ax.figure.canvas.mpl_connect('scroll_event', self._on_scroll)

    def _on_click(self, event: MouseEvent):
        """Handles mouse clicks to add points or close the polygon."""
        if event.inaxes != self.ax or self.is_closed or event.xdata is None:
            return

        # Left click to add a point
        if event.button == 1:
            self.polygon_points.append((event.xdata, event.ydata))
            self._update_line()

        # Right click to close the polygon
        elif event.button == 3 and len(self.polygon_points) >= 3:
            self.is_closed = True
            self.polygon_points.append(self.polygon_points[0]) # Append start point to end
            self._update_line()
            print("Polygon closed. Press [Enter] to save and continue.")

    def _on_scroll(self, event: ScrollEvent):
        """Handles mouse scroll events for zooming."""
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None: return # Return if scroll is outside axes

        base_scale = 1.2
        scale_factor = base_scale if event.button == 'down' else 1 / base_scale

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (x - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rely = (y - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([x - new_width * relx, x + new_width * (1 - relx)])
        self.ax.set_ylim([y - new_height * rely, y + new_height * (1 - rely)])
        self.ax.figure.canvas.draw_idle()

    def _update_line(self):
        """Redraws the polygon line on the canvas."""
        if not self.polygon_points:
            self.line.set_data([], [])
        else:
            xs, ys = zip(*self.polygon_points)
            self.line.set_data(xs, ys)
        self.ax.figure.canvas.draw_idle()

    def get_polygon(self) -> List[Point]:
        """Returns the final list of points if the polygon is closed."""
        return self.polygon_points if self.is_closed else []

class MaskCreator:
    """
    Main application class to orchestrate the process of creating binary masks
    from images by drawing polygons.
    """
    def __init__(self, input_dir: Path, output_dir: Path):
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        supported = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        self.image_files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in supported])

        if not self.image_files:
            print(f"Warning: No images found in {input_dir}")

        self.current_image_path: Path
        self.img_shape: Tuple[int, int, int]
        self.selector: RegionSelector

    def run(self):
        """Main loop to process each image in the input directory."""
        print(f"Starting process for {len(self.image_files)} images...")
        for img_path in self.image_files:
            self.current_image_path = img_path
            self._process_image()
        print("\nAll images processed. Exiting.")

    def _process_image(self):
        """Loads a single image and opens the interactive selection window."""
        try:
            img = plt.imread(self.current_image_path)
            self.img_shape = img.shape
        except Exception as e:
            print(f"Error loading {self.current_image_path.name}: {e}. Skipping.")
            return

        fig, ax = plt.subplots(figsize=(16, 12))
        ax.imshow(img)
        ax.set_title("L-click: add point | R-click: close polygon | Scroll: zoom | Enter: save | Esc: skip")
        ax.axis('off')

        self.selector = RegionSelector(ax)
        fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        plt.show() # Blocks until the window is closed

    def _on_key_press(self, event: KeyEvent):
        """Handles key press events for saving or skipping."""
        if event.key == 'enter':
            polygon = self.selector.get_polygon()
            if polygon:
                self._save_mask(polygon)
                plt.close(event.canvas.figure)
            else:
                print("Polygon is not closed. Please right-click to close it first.")

        elif event.key == 'escape':
            print(f"Skipping {self.current_image_path.name}.")
            plt.close(event.canvas.figure)

    def _save_mask(self, polygon: List[Point]):
        """Creates and saves a binary mask from the given polygon."""
        mask = np.zeros((self.img_shape[0], self.img_shape[1]), dtype=np.uint8)
        path = Path(polygon)

        # Create a grid of coordinates
        x, y = np.meshgrid(np.arange(self.img_shape[1]), np.arange(self.img_shape[0]))
        points = np.vstack((x.flatten(), y.flatten())).T

        # Check which points fall inside the polygon path
        mask_indices = path.contains_points(points).reshape(self.img_shape[0], self.img_shape[1])
        mask[mask_indices] = 255  # Set points inside the polygon to white

        # Construct the output path
        save_path = self.output_dir / f"{self.current_image_path.stem}_mask.png"

        try:
            Image.fromarray(mask).save(save_path)
            print(f"Successfully saved mask to: {save_path}")
        except Exception as e:
            print(f"Error saving mask: {e}")

def main():
    parser = argparse.ArgumentParser(description="Create binary masks by drawing polygons on images.")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Directory containing source images.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Directory to save the mask files.")
    args = parser.parse_args()

    try:
        creator = MaskCreator(args.input, args.output)
        creator.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
