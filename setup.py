"""Setup script for ML Nutri-Score Prediction project."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="nutriscore-ml",
    version="0.1.0",
    author="Jacopo Parretti, Cesare Fraccaroli",
    author_email="jacopo.parretti@studenti.univr.it, cesare.fraccaroli@studenti.univr.it",
    description="Multi-class classification of food product Nutri-Scores using classical ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/djacoo/ml-nutriscore-prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nutriscore-preprocess=scripts.preprocess_data:main",
            "nutriscore-train=scripts.train_models:main",
        ],
    },
)
