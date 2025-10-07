import argparse
from pathlib import Path
from typing import List

from rembg import remove
from tqdm import tqdm

def process_image(input_path: Path, output_path: Path):
    """
    Reads an image, removes its background, and saves it to the output path.

    Args:
        input_path (Path): The path to the input image file.
        output_path (Path): The path where the processed image will be saved.
    """
    try:
        with open(input_path, 'rb') as f_in:
            input_bytes = f_in.read()
        
        output_bytes = remove(input_bytes)
        
        with open(output_path, 'wb') as f_out:
            f_out.write(output_bytes)
    
    except Exception as e:
        # Print error but continue with the batch
        print(f"\nError processing {input_path.name}: {e}. Skipping this file.")

def batch_remove_background(
    input_dir: Path, 
    output_dir: Path, 
    to_png: bool = False
):
    """
    Processes all images in a given directory to remove their backgrounds.

    Args:
        input_dir (Path): The directory containing the source images.
        output_dir (Path): The directory where processed images will be saved.
        to_png (bool): If True, converts all output images to PNG format.
    """
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all supported image files
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
    image_files: List[Path] = [
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in supported_formats
    ]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images. Starting background removal...")
    
    # Process files with a progress bar
    for file_path in tqdm(image_files, desc="Processing images"):
        output_filename = file_path.stem + ".png" if to_png else file_path.name
        output_path = output_dir / output_filename
        
        process_image(file_path, output_path)

def main():
    """Main function to parse command-line arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="A command-line tool to remove the background from all images in a folder."
    )
    parser.add_argument(
        "-i", "--input", 
        type=Path, 
        required=True, 
        help="Path to the input folder containing images."
    )
    parser.add_argument(
        "-o", "--output", 
        type=Path, 
        required=True, 
        help="Path to the output folder where processed images will be saved."
    )
    parser.add_argument(
        "--png",
        action='store_true',
        help="Force output images to be saved in PNG format to ensure transparency support."
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"Error: Input path '{args.input}' is not a valid directory.")
        return

    batch_remove_background(args.input, args.output, args.png)
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
