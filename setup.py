from setuptools import setup, find_packages

setup(
    name="cytobulk",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="CytoBulk: A comprehensive toolkit for spatial transcriptomics analysis",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/kristaxying/CytoBulk",
    packages=find_packages(),
    python_requires=">=3.10",
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3.10",
    ],
)
