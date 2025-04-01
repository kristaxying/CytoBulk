import openslide
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


def process_svs_image(svs_path, output_dir, crop_size=224, magnification=1, 
                      center_x=None, center_y=None, fold_width=10, fold_height=10):
    """
    Process an SVS (Whole Slide Image) file to crop a specific region, resize it, 
    and save smaller tiles extracted from the region.

    Parameters
    ----------
    svs_path : string
        Path to the SVS file to be processed.
    output_dir : string
        Path to the directory where cropped tiles will be saved.
    crop_size : int, optional
        Size of each cropped tile (default is 224x224 pixels).
    magnification : int, optional
        Magnification factor for the cropped region (default is 1).
    center_x : int, optional
        X-coordinate of the cropping center. Defaults to the image center if None.
    center_y : int, optional
        Y-coordinate of the cropping center. Defaults to the image center if None.
    fold_width : int, optional
        Number of tiles in the horizontal direction (default is 10).
    fold_height : int, optional
        Number of tiles in the vertical direction (default is 10).
        
    Returns
    -------
    None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the SVS file
    slide = openslide.OpenSlide(svs_path)

    # Get the dimensions of the whole slide image
    width, height = slide.dimensions
    print(f"Original image size: {width}x{height}")

    # Set the cropping center to the image center if not provided
    if center_x is None:
        center_x = width // 2
    if center_y is None:
        center_y = height // 2
    print(f"Image center: ({center_x}, {center_y})")

    # Calculate the dimensions of the cropping region
    crop_width = fold_width * crop_size  # Total width of the cropped region
    crop_height = fold_height * crop_size  # Total height of the cropped region

    # Calculate the starting coordinates of the cropping region
    start_x = max(center_x - crop_width // 2, 0)  # Ensure it doesn't go out of bounds
    start_y = max(center_y - crop_height // 2, 0)  # Ensure it doesn't go out of bounds
    print(f"Crop region: Start=({start_x}, {start_y}), Size=({crop_width}, {crop_height})")

    # Read the cropped region from the slide at level 0 (highest resolution)
    region = slide.read_region((start_x, start_y), 0, (crop_width, crop_height))
    region = region.convert("RGB")  # Convert the cropped region to RGB format

    # Visualize the cropped region using matplotlib
    plt.imshow(region)
    plt.title("Cropped Region")
    plt.axis("off")
    plt.show()

    # Calculate the enlarged dimensions of the cropped region
    enlarged_width = crop_width * magnification
    enlarged_height = crop_height * magnification

    # Resize the cropped region to the specified magnification
    region_enlarged = region.resize((enlarged_width, enlarged_height), Image.LANCZOS)
    print(f"Enlarged image size: {region_enlarged.size}")

    # Loop through the enlarged region and save tiles of size (crop_size x crop_size)
    for i in range(fold_width):  # Loop through horizontal tiles
        for j in range(fold_height):  # Loop through vertical tiles
            # Calculate the coordinates for the current tile
            x = i * crop_size
            y = j * crop_size

            # Ensure the tile is within the bounds of the enlarged image
            if x + crop_size <= enlarged_width and y + crop_size <= enlarged_height:
                # Crop the tile from the enlarged region
                cropped = region_enlarged.crop((x, y, x + crop_size, y + crop_size))

                # Generate the filename for the tile
                folder_name = f"{i}_{j}/"
                full_path = os.path.join(output_dir, folder_name)


                os.makedirs(full_path, exist_ok=True)

                filename = f"0.jpg"

                # Save the tile to the output directory
                cropped.save(os.path.join(full_path,filename))