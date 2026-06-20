# Format: MAJOR.MINOR.PATCH
__version__ = "1.0.1"

# Additional version info
__author__ = "Carolin Schwitalla"
__copyright__ = "Copyright 2024"
__license__ = "MIT"
__description__ = "3D-UNet for cell nuclei segmentation"
__url__ = "https://github.com/CaroAMN/numorph_3dunet"

# Package requirements (corresponds to pyproject.toml)
__requires__ = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "tensorflow>=2.8.0",
    "scikit-image>=0.19.0",
    "opencv-python>=4.5.0",
    "mat73>=0.59"
]

# Optional: version string for display
VERSION_STRING = f"""
numorph_3dunet v{__version__}
Author: {__author__}
License: {__license__}
"""