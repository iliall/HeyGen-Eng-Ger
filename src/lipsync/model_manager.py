import os
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib


MODEL_URLS = {
    'wav2lip': {
        'url': 'https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip.pth',
        'filename': 'wav2lip.pth',
        'md5': None,  # Optional: add MD5 checksum for verification
        'gdrive_id': '1tLZP-4wHxJ6I9y7K1L9h3y0b3n7c8k9j'  # Google Drive ID as backup
    },
    'wav2lip_gan': {
        'url': 'https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip_gan.pth',
        'filename': 'wav2lip_gan.pth',
        'md5': None
    },
    's3fd': {
        'url': 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth',
        'filename': 's3fd.pth',
        'md5': None
    }
}


def get_model_path(model_name: str) -> Path:
    """
    Get the path where a model should be stored.

    Args:
        model_name: Name of the model (wav2lip, wav2lip_gan, s3fd)

    Returns:
        Path to model file
    """
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / 'models' / 'lipsync'
    models_dir.mkdir(parents=True, exist_ok=True)

    if model_name not in MODEL_URLS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_URLS.keys())}")

    return models_dir / MODEL_URLS[model_name]['filename']


def download_file(url: str, destination: Path, desc: str = "Downloading") -> None:
    """
    Download a file with progress bar.

    Args:
        url: URL to download from
        destination: Path to save file
        desc: Description for progress bar
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(destination, 'wb') as f, tqdm(
        desc=desc,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def verify_md5(file_path: Path, expected_md5: str) -> bool:
    """
    Verify MD5 checksum of a file.

    Args:
        file_path: Path to file
        expected_md5: Expected MD5 hash

    Returns:
        True if checksum matches, False otherwise
    """
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)

    return md5_hash.hexdigest() == expected_md5


def download_model(model_name: str, force_download: bool = False) -> Path:
    """
    Download a pre-trained model if it doesn't exist.

    Args:
        model_name: Name of the model to download (wav2lip, wav2lip_gan, s3fd)
        force_download: Force re-download even if file exists

    Returns:
        Path to downloaded model file

    Raises:
        ValueError: If model name is unknown
        requests.HTTPError: If download fails
    """
    if model_name not in MODEL_URLS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_URLS.keys())}")

    model_info = MODEL_URLS[model_name]
    model_path = get_model_path(model_name)

    # Check if model already exists
    if model_path.exists() and not force_download:
        print(f"Model {model_name} already exists at: {model_path}")

        # Verify checksum if provided
        if model_info['md5']:
            if verify_md5(model_path, model_info['md5']):
                print(f"Checksum verified for {model_name}")
            else:
                print(f"Checksum mismatch for {model_name}, re-downloading...")
                force_download = True

        if not force_download:
            return model_path

    # Download the model
    print(f"Downloading {model_name} from {model_info['url']}...")
    try:
        download_file(model_info['url'], model_path, desc=f"Downloading {model_name}")
        print(f"Downloaded {model_name} to: {model_path}")

        # Verify checksum if provided
        if model_info['md5']:
            if verify_md5(model_path, model_info['md5']):
                print(f"Checksum verified for {model_name}")
            else:
                raise ValueError(f"Downloaded file checksum does not match for {model_name}")

        return model_path

    except Exception as e:
        # Clean up partial download
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(f"Failed to download {model_name}: {e}")


def download_all_models(force_download: bool = False) -> dict:
    """
    Download all required models for lip-sync.

    Args:
        force_download: Force re-download even if files exist

    Returns:
        Dictionary mapping model names to their paths
    """
    models = {}

    print("Downloading required models for lip-sync...")
    print("This may take a few minutes (total ~400MB)...\n")

    for model_name in ['wav2lip', 's3fd']:
        try:
            models[model_name] = download_model(model_name, force_download)
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")
            raise

    print("\nAll models downloaded successfully!")
    return models


def check_models_exist() -> bool:
    """
    Check if all required models are downloaded.

    Returns:
        True if all models exist, False otherwise
    """
    required_models = ['wav2lip', 's3fd']

    for model_name in required_models:
        model_path = get_model_path(model_name)
        if not model_path.exists():
            return False

    return True


if __name__ == "__main__":
    # Download all models when run directly
    download_all_models()
