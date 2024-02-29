from setuptools import setup, find_packages

setup(
    name="ChemIC",
    version="1.2",
    description="Chemical images classification project. Program for training the neural network model and web service for classification images",
    author="Dr.Aleksei Krasnov",
    author_email="a.krasnov@digital-science.com",
    license="MIT",
    python_requires=">=3.10,<3.12",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    url="https://github.com/ontochem/ChemIC.git",
    packages=find_packages(exclude=["tests", "tests.*", "models", "Benchmark"]),
    package_dir={'chemic': 'chemic'},
    install_requires=[
        "flask>=3.0.0",
        "gunicorn>=21.2.0",
        "numpy>=1.26.3",
        "pandas>=2.2.0",
        "pillow>=10.2.0",
        "requests>=2.31.0",
        "scikit-learn>=1.3.2",
        "torch>=2.2.0",
        "torchmetrics>=1.2.1",
        "torchvision>=0.17.0",
    ],
)
