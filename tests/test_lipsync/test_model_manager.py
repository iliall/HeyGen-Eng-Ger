import pytest
from pathlib import Path
from src.lipsync.model_manager import (
    get_model_path,
    check_models_exist,
    MODEL_URLS
)


class TestModelManager:
    """Test model download and management functionality."""

    def test_get_model_path_wav2lip(self):
        """Test getting path for wav2lip model."""
        model_path = get_model_path('wav2lip')

        assert isinstance(model_path, Path)
        assert model_path.name == 'wav2lip.pth'
        assert 'models/lipsync' in str(model_path)

    def test_get_model_path_s3fd(self):
        """Test getting path for s3fd model."""
        model_path = get_model_path('s3fd')

        assert isinstance(model_path, Path)
        assert model_path.name == 's3fd.pth'
        assert 'models/lipsync' in str(model_path)

    def test_get_model_path_invalid(self):
        """Test getting path for invalid model raises error."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_path('invalid_model')

    def test_model_urls_exist(self):
        """Test that all model URLs are defined."""
        assert 'wav2lip' in MODEL_URLS
        assert 's3fd' in MODEL_URLS

        # Check structure
        for model_name, info in MODEL_URLS.items():
            assert 'url' in info
            assert 'filename' in info
            assert info['url'].startswith('http')
            assert info['filename'].endswith(('.pth', '.pt'))

    def test_check_models_exist(self):
        """Test checking if models exist."""
        # This will return False initially (models not downloaded)
        # or True if models are already downloaded
        result = check_models_exist()
        assert isinstance(result, bool)

    def test_model_path_creates_directory(self):
        """Test that getting model path creates the directory."""
        model_path = get_model_path('wav2lip')

        # Directory should be created
        assert model_path.parent.exists()
        assert model_path.parent.is_dir()


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
