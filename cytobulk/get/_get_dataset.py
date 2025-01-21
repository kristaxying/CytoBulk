import os
import requests
from pkg_resources import resource_filename

def ensure_file_exists(file_dir, file_name, download_url):
    """
    Ensure the specified file exists. If it doesn't, download it from the provided URL.

    :param file_dir: The directory where the file should be located.
    :param file_name: The name of the file to check or download.
    :param download_url: The URL to download the file from.
    """
    # Combine directory and file name to create the full file path
    file_path = os.path.join(file_dir, file_name)

    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
    else:
        print(f"File does not exist. Downloading from: {download_url}")
        
        # Ensure the directory exists
        os.makedirs(file_dir, exist_ok=True)
        
        # Download the file
        try:
            response = requests.get(download_url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"File successfully downloaded and saved to: {file_path}")
        except requests.exceptions.RequestException as e:
            print(f"File download failed: {e}")
            raise

