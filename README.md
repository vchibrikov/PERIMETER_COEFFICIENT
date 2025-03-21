# PERIMETER_COEFFICIENT
- Current repository provides an image processing script that performs edge detection, object filtering, and calculates the lengths of perpendiculars drawn from midpoints of object boundaries in images. The program can also plot the filtered objects, midpoint lines, and tilted perpendiculars, with interactive sliders to adjust threshold values for edge detection.
- Repository is created for the project entitled "Printing of 3D biomaterials inspired by plant cell wall", supported by the National Science Centre, Poland (grant nr - 2023/49/B/NZ9/02979).
- Research methodology is an automative approach reported in papers:

Merli, M., Sardelli, L., Baranzini, N., Grimaldi, A., Jacchetti, E., Raimondi, M. T., ... & Tunesi, M. (2022). Pectin-based bioinks for 3D models of neural tissue produced by a pH-controlled kinetics. Frontiers in Bioengineering and Biotechnology, 10, 1032542.
Gillispie, G., Prim, P., Copus, J., Fisher, J., Mikos, A. G., Yoo, J. J., ... & Lee, S. J. (2020). Assessment methodologies for extrusion-based bioink printability. Biofabrication, 12(2), 022003.
Sardelli, L., Tunesi, M., Briatico-Vangosa, F., & Petrini, P. (2021). 3D-Reactive printing of engineered alginate inks. Soft Matter, 17(35), 8105-8117.

Current repository is composed fro three principal blocks:
- 1_BACKGROUND_REMOVER.py (remove background from image)
- 2_PERIMETER_COEFFICIENT.py (calculate object lengths and widths)
- 3_SCALE.py (collect data on scale)

## Requirements

The following Python libraries are required:

- os
- rembg
- PIL
- io
- OpenCV `cv2`
- numpy
- matplotlib
- math
- openpyxl
- pandas

## Description

### 1_BACKGROUND_REMOVER.py
This Python script removes the background from images in a specified input folder and saves the processed images to an output folder. It leverages the `rembg` library to perform background removal and works with common image formats like `.png`, `.jpg`, `.jpeg`, and `.bmp`. Examples of raw and processed images are shown on Fig.1 and Fig.2, respectively.

![Fig_1-fotor-2025032117652](https://github.com/user-attachments/assets/b3fb98d5-8fc1-494d-8e41-8a9b3aa2f6de)
Fig.1. Example of raw image.

![Fig_2](https://github.com/user-attachments/assets/89017ff4-a092-4c80-afab-7d19368bbdf5)
Fig.2. Example of processed image.

### 2_PERIMETER_COEFFICIENT.py
This Python script processes a collection of images by applying Canny edge detection, Gaussian blur, morphological closing, and contour filling to detect and analyze objects within the images. Using the centroid of detected contours, the code calculates distances to edge points and perpendiculars, visualizing results with overlaid blue and red lines. Additionally, it computes and stores the lengths of these perpendiculars, saving the results in Excel files for each image. The script features interactive sliders for fine-tuning parameters like Canny thresholds, kernel size, and blur, making it suitable for detailed image analysis and measurements. Exa,ple of processed data is shown on Fig.3.

![Fig_3](https://github.com/user-attachments/assets/b23ba92c-6343-4485-85d8-f5fc0a834910)
Fig.3. Example of a processed image.

### 3_SCALE.py
This code is designed for processing images in a folder, allowing the user to select two points on each image, calculate the pixel distance between those points, and save the results in an Excel file. It uses Matplotlib for visualization and interaction, OpenCV for image handling, and Pandas for saving results.

## Notes
Ensure that the images you are using are of good quality with clear object boundaries for better edge detection results.
The program can be adapted to process a larger variety of images by adjusting the parameters.

## License
This code is licensed under the MIT License. See the LICENSE file for more details.
