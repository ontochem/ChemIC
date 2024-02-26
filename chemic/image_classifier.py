"""Chemical Images Classifier Module:
This module provides image classification functionality.
It uses pre-trained models for classifying chemical images.

Dependencies:
    - concurrent
    - pathlib
    - typing
    - flask
    - torch
    - torchvision
    - config (assuming Config class is defined in the 'config' module)
    - loading_images (assuming MixedImagesDataset class is defined in the 'loading_images' module)

Usage:
    Instantiate the ImageClassifier class and utilize the 'send_to_classifier' method to start classification process.

Author:
    Dr. Aleksei Krasnov
    a.krasnov@digital-science.com
    Date: February 26, 2024
"""

import importlib.metadata
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, List, NamedTuple, Union
from flask import jsonify, Response
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from .config import Config
from .loading_images import MixedImagesDataset


# Define the transformation for the images
transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.Grayscale(num_output_channels=3),  # Convert to RGB if grayscale
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Define class labels. Order of class label in the NamedTuple is essential!
class ChemicalLabels(NamedTuple):
    """"Class label for image classifier"""
    single_chemical_structure: str
    chemical_reactions: str
    no_chemical_structures: str
    multiple_chemical_structures: str


# Creating an instance of ChemicalLabels
chem_labels = ChemicalLabels(single_chemical_structure='single chemical structure',
                             chemical_reactions='chemical reactions',
                             no_chemical_structures='no chemical structures',
                             multiple_chemical_structures='multiple chemical structures')

# Load ML models
classifier_model = Config.get_models()


class ImageClassifier:
    """
    A class encapsulating image classification functionality.
    """
    def __init__(self) -> None:
        """Initializes the ImageRecognizer instance with queues.
        """
        self.mixed_loader = None
        self.results = []  # Store results of recognition in a list

    def send_to_classifier(self, image_path: str) -> Union[Tuple[Response, int], List]:
        """
        Enqueues images for classification based on the provided image path.
        Parameters:
            - image_path (str): Path to the image file or directory.
        Returns:
            - Tuple[Response, int]: Success message or Error response if the image path is invalid.
        """
        try:
            # Create a DataLoader for the mixed images
            mixed_dataset = MixedImagesDataset(path_or_dir=image_path, transform=transform)
            self.mixed_loader = DataLoader(mixed_dataset, batch_size=1, shuffle=False, num_workers=0)
        except Exception as e:
            print(f"Exception: {e} {image_path}")
            result_entry = {
                'image_id': image_path,
                'predicted_label': 'Error! File is not an image',
                            }
            self.results.append(result_entry)
            return self.results
        else:
            self.process_images()
            return jsonify({"message": "Images have been classified and put to queues."}), 202

    def process_images(self) -> None:
        """
        Processes images in the mixed_loader using multithreading.
        The images are processed concurrently using a ThreadPoolExecutor with a maximum number of worker threads
        determined by min of the CPU count or number of images in self.mixed_loader.
        """
        prog_flag = 'ChemIC'
        prog_version = self.get_package_version(prog_flag)
        with ThreadPoolExecutor(max_workers=min((os.cpu_count()), len(self.mixed_loader))) as executor:
            futures = [executor.submit(self.process_image, image_data_) for image_data_ in self.mixed_loader]
            for future in as_completed(futures):
                image_path, predicted_label = future.result()
                print(image_path, predicted_label)
                result_entry = {
                    'image_id': Path(image_path).name,
                    'predicted_label': predicted_label,
                    'program': prog_flag,
                    'program_version': prog_version
                }
                self.results.append(result_entry)

    @staticmethod
    def process_image(image_data):
        """
        Processes a single image in the mixed_loader and returns the image path and predicted class label by
        using chemical images classifier.

        Parameters:
        - image_data (Tuple[str, torch.Tensor]): A tuple containing the image path and the corresponding image tensor.

        Returns:
        - Tuple[str, str]: A tuple containing the image path and the predicted class label.
        """
        image_path, image = image_data
        image_path = image_path[0]  # Extract the image path from the batch
        try:
            with torch.no_grad():
                output = classifier_model(image)
                _, predicted = torch.max(output.data, 1)
                return image_path, chem_labels[predicted.item()]
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    @staticmethod
    def get_package_version(package_name):
        """
        Get the version of the specified Python package.

        Parameters:
            package_name (str): The name of the Python package.

        Returns:
            str: The version of the specified package if installed.
                 If the package is not installed, returns a message indicating that the package is not installed.
        """
        try:
            package_version = importlib.metadata.version(package_name)
            return package_version
        except importlib.metadata.PackageNotFoundError:
            return f"{package_name} is not installed"
