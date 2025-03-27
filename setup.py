from setuptools import setup, find_packages

setup(
    name="CytoBulk",  
    version="0.1.10",  
    author="Xueying WANG",  
    author_email="your_email@example.com",  
    description="Integrating transcriptional data to decipher the tumor microenvironment with the graph frequency domain model",  
    long_description=open("README.md", encoding="utf-8").read(),  
    long_description_content_type="text/markdown", 
    url="https://github.com/kristaxying/CytoBulk",  
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={
        "cytobulk": ["preprocessing/*.R"],  # 指定需要打包的 .R 文件
    },
    classifiers=[
        "Programming Language :: Python :: 3",  
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",  
    install_requires=[
        "anndata>=0.10.0",
        "cellpose>=3.0.10",
        "imageio",
        "matplotlib",
        "numpy>=1.23.0",
        "openslide-python",
        "ortools==9.3.10497",
        "pandas>=2.2.0",
        "Pillow",
        "POT==0.9.5",
        "rpy2>=3.5.0",
        "scanpy",
        "scikit-learn",
        "scipy",
        "seaborn",
        "torch>2.1.0",
        "torchvision",
        "tqdm",
        "requests",
        "openslide-python",
        "openslide-bin",
        "scikit-image",
        "igraph",
        "leidenalg"
    ],
)