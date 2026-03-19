from setuptools import setup, find_packages

setup(
    name="ceiba",
    version="0.1.0",
    author="Gabriel Hanson",
    author_email="",                        # add if desired
    description="Spatial transcriptomics methods for spatial analysis pipelines",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gabehanson/ceiba",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.12",
    install_requires=[
        # core data
        "numpy",
        "pandas",
        # single-cell / spatial
        "anndata",
        "scanpy",
        "squidpy",
        # ml / stats
        "scikit-learn",
        "scipy",
        # visualization
        "matplotlib",
        "seaborn",
        # utilities
        "tqdm",
        # opls
        "pyopls",
    ],
    extras_require={
        "dev": [
            "pytest",
            "jupyter",
            "nbstripout",
            "black",
            "isort",
        ],
        "utag": [
            "utag",               # optional — install separately if used
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)

# To install locally in editable mode, run from the repo root:
#   pip install -e .
# Your notebooks can then do:
#   from ceiba import ...