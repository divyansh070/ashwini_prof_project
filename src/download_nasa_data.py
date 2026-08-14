import os
import sys
import logging
import urllib.request
import zipfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [NASADownload] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NASADownload")

URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
RAW_DIR = "data/raw/nasa_battery"

def download_and_extract():
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, "BatteryDataset.zip")
    
    if not os.path.exists(zip_path):
        logger.info(f"Downloading NASA Battery Dataset from {URL}...")
        try:
            import subprocess
            subprocess.run(["curl", "-L", "-o", zip_path, URL], check=True)
            logger.info("Download complete.")
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            sys.exit(1)
    else:
        logger.info(f"Zip file already exists at {zip_path}.")

    logger.info("Extracting files...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
        logger.info(f"Extraction complete to {RAW_DIR}.")
    except Exception as e:
        logger.error(f"Failed to extract zip file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_and_extract()
