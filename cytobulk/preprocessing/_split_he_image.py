import openslide
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


def process_svs_image(svs_path, output_dir, crop_size=224, magnification=1, 
                      center_x=None, center_y=None, fold_width=10, fold_height=10,
                      enable_cropping=True):
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
        Only used when enable_cropping=True.
    center_y : int, optional
        Y-coordinate of the cropping center. Defaults to the image center if None.
        Only used when enable_cropping=True.
    fold_width : int, optional
        Number of tiles in the horizontal direction (default is 10).
        When enable_cropping=False, this determines the tile size for the entire image.
    fold_height : int, optional
        Number of tiles in the vertical direction (default is 10).
        When enable_cropping=False, this determines the tile size for the entire image.
    enable_cropping : bool, optional
        Whether to crop a specific region (True) or process the entire image (False).
        Default is True.
        
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

    if enable_cropping:
        # Cropping mode: extract specified region
        print("Mode: Cropping enabled - processing specific region")
        
        # Set the cropping center to the image center if not provided
        if center_x is None:
            center_x = width // 2
        if center_y is None:
            center_y = height // 2
        print(f"Cropping center: ({center_x}, {center_y})")

        # Calculate the dimensions of the cropping region
        crop_width = fold_width * crop_size  # Total width of the cropped region
        crop_height = fold_height * crop_size  # Total height of the cropped region

        # Calculate the starting coordinates of the cropping region
        start_x = max(center_x - crop_width // 2, 0)  # Ensure it doesn't go out of bounds
        start_y = max(center_y - crop_height // 2, 0)  # Ensure it doesn't go out of bounds
        
        # Ensure the crop region doesn't exceed image boundaries
        if start_x + crop_width > width:
            start_x = width - crop_width
        if start_y + crop_height > height:
            start_y = height - crop_height
            
        print(f"Crop region: Start=({start_x}, {start_y}), Size=({crop_width}, {crop_height})")

        # Read the cropped region from the slide at level 0 (highest resolution)
        region = slide.read_region((start_x, start_y), 0, (crop_width, crop_height))
        region = region.convert("RGB")

        # Visualize the processed region using matplotlib
        plt.figure(figsize=(10, 8))
        plt.imshow(region)
        plt.title(f"Cropped Region (Center: {center_x}, {center_y})")
        plt.axis("off")
        plt.show()

        # Calculate the enlarged dimensions of the region
        enlarged_width = region.width * magnification
        enlarged_height = region.height * magnification

        # Resize the region to the specified magnification
        region_enlarged = region.resize((enlarged_width, enlarged_height), Image.LANCZOS)
        print(f"Enlarged image size: {region_enlarged.size}")

        # Loop through the enlarged region and save tiles of size (crop_size x crop_size)
        saved_tiles = 0
        for i in range(fold_width):  # Loop through horizontal tiles
            for j in range(fold_height):  # Loop through vertical tiles
                # Calculate the coordinates for the current tile
                x = i * crop_size
                y = j * crop_size

                # Ensure the tile is within the bounds of the enlarged image
                if x + crop_size <= enlarged_width and y + crop_size <= enlarged_height:
                    # Crop the tile from the enlarged region
                    cropped = region_enlarged.crop((x, y, x + crop_size, y + crop_size))

                    # Generate the folder name for the tile
                    folder_name = f"{x}_{y}/"
                    full_path = os.path.join(output_dir, folder_name)

                    # Create the directory if it doesn't exist
                    os.makedirs(full_path, exist_ok=True)

                    # Generate the filename for the tile
                    filename = f"0.jpg"

                    # Save the tile to the output directory
                    cropped.save(os.path.join(full_path, filename))
                    saved_tiles += 1
                else:
                    print(f"Skipping tile ({i}, {j}) - out of bounds")

        print(f"Processing completed! Saved {saved_tiles} tiles to {output_dir}")

    else:
        # Non-cropping mode: process the entire image by tiling
        print("Mode: Cropping disabled - processing entire image with tiling")
        
        # Calculate how many tiles we can fit in the entire image
        tiles_per_row = width // crop_size
        tiles_per_col = height // crop_size
        
        print(f"Image can be divided into: {tiles_per_row} x {tiles_per_col} tiles")
        print(f"Each tile size: {crop_size} x {crop_size} pixels")
        
        # If fold_width and fold_height are specified, use them to limit the number of tiles
        # Otherwise, process all possible tiles
        max_tiles_x =tiles_per_row
        max_tiles_y = tiles_per_col
        
        print(f"Processing {max_tiles_x} x {max_tiles_y} tiles")
        
        # Show a preview of the image (downsampled for display)
        try:
            # Get a downsampled version for preview
            preview_level = min(2, slide.level_count - 1)  # Use level 2 or highest available
            preview_size = slide.level_dimensions[preview_level]
            preview_image = slide.read_region((0, 0), preview_level, preview_size)
            preview_image = preview_image.convert("RGB")
            
            plt.figure(figsize=(12, 8))
            plt.imshow(preview_image)
            plt.title(f"Full Image Preview (Level {preview_level})")
            plt.axis("off")
            
            # Overlay grid lines to show tile boundaries (scaled to preview)
            scale_x = preview_size[0] / width
            scale_y = preview_size[1] / height
            
            # Draw vertical lines
            for i in range(1, max_tiles_x):
                x_line = i * crop_size * scale_x
                plt.axvline(x=x_line, color='red', alpha=0.5, linewidth=1)
            
            # Draw horizontal lines  
            for j in range(1, max_tiles_y):
                y_line = j * crop_size * scale_y
                plt.axhline(y=y_line, color='red', alpha=0.5, linewidth=1)
                
            plt.show()
        except Exception as e:
            print(f"Could not generate preview: {e}")

        # Process tiles across the entire image
        saved_tiles = 0
        total_tiles = max_tiles_x * max_tiles_y
        
        print(f"Starting to process {total_tiles} tiles...")
        
        for i in range(max_tiles_x):  # Loop through horizontal tiles
            for j in range(max_tiles_y):  # Loop through vertical tiles
                # Calculate the coordinates for the current tile in the original image
                x = i * crop_size
                y = j * crop_size

                try:
                    # Read the tile directly from the slide at level 0 (highest resolution)
                    tile_region = slide.read_region((x, y), 0, (crop_size, crop_size))
                    tile_region = tile_region.convert("RGB")
                    
                    # Apply magnification if specified
                    if magnification != 1:
                        new_size = (crop_size * magnification, crop_size * magnification)
                        tile_region = tile_region.resize(new_size, Image.LANCZOS)

                    # Generate the folder name for the tile
                    folder_name = f"{x}_{y}/"
                    full_path = os.path.join(output_dir, folder_name)

                    # Create the directory if it doesn't exist
                    os.makedirs(full_path, exist_ok=True)

                    # Generate the filename for the tile
                    filename = f"0.jpg"

                    # Save the tile to the output directory
                    tile_region.save(os.path.join(full_path, filename))
                    saved_tiles += 1
                    
                    # Print progress every 100 tiles
                    if saved_tiles % 100 == 0:
                        print(f"Processed {saved_tiles}/{total_tiles} tiles ({saved_tiles/total_tiles*100:.1f}%)")
                        
                except Exception as e:
                    print(f"Error processing tile ({i}, {j}): {e}")

        print(f"Processing completed! Saved {saved_tiles} tiles to {output_dir}")
        print(f"Tiles are organized in {max_tiles_x} columns x {max_tiles_y} rows")
        
        # Save processing information
        info_file = os.path.join(output_dir, "processing_info.txt")
        with open(info_file, 'w') as f:
            f.write(f"SVS Processing Information\n")
            f.write(f"========================\n\n")
            f.write(f"Source file: {svs_path}\n")
            f.write(f"Original image size: {width} x {height}\n")
            f.write(f"Tile size: {crop_size} x {crop_size}\n")
            f.write(f"Magnification: {magnification}\n")
            f.write(f"Grid size: {max_tiles_x} x {max_tiles_y}\n")
            f.write(f"Total tiles saved: {saved_tiles}\n")
            f.write(f"Coverage area: {max_tiles_x * crop_size} x {max_tiles_y * crop_size} pixels\n")
        
        print(f"Processing information saved to: {info_file}")
    
    # Close the slide to free up resources
    slide.close()