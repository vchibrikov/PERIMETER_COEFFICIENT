import os
from rembg import remove
from PIL import Image
import io

# Function to process images in a folder
def remove_background_from_folder(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop over each file in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if os.path.isfile(input_path):
            # Check if the file is an image (e.g., .jpg, .jpeg, .png)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                # Open the image
                with open(input_path, 'rb') as input_file:
                    input_image = input_file.read()

                # Remove the background
                output_image = remove(input_image)

                # Write the processed image to the output folder
                output_path = os.path.join(output_folder, filename)

                # Save the output image (in the same format as input)
                with open(output_path, 'wb') as output_file:
                    output_file.write(output_image)

                print(f"Processed and saved: {filename}")

# Specify the input and output folder paths
input_folder = 'path/to/input/folder'  # Replace with the path to your input folder
output_folder = 'path/to/output/folder'  # Replace with the path to your output folder

# Call the function to process the images
remove_background_from_folder(input_folder, output_folder)
