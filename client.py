"""Chemical Images Classification Client Module:
This module facilitates the interaction between the client and the ChemIC web-server.
The client sends to server the path to an image file or directory, and the server processes the images,
providing the client with the recognition results.

Dependencies:
    - argparse
    - datetime
    - requests
    - pandas
    - time

Usage:
    Run this module with the required arguments:
        python client.py --image_path /path/to/images --export_dir /path/to/export

    Optional arguments:
        --export_dir: Export directory for the recognition results in the csv format. Default is the current directory.

Author:
    Dr. Aleksei Krasnov
    a.krasnov@digital-science.com
    Date: February 26, 204
"""

import argparse
import datetime
import time
from pathlib import Path

import pandas as pd
import requests

start = time.time()


class ChemClassifierClient:
    """
    A client for interacting with the ChemIC application's server.

    Attributes:
        server_url (str): The URL of the ChemICR server.

    Methods:
        classify(image_path: str) -> dict:
            Sends a POST request to the server with the specified image path or directory.
            Returns the classification results in dictionary format.

        healthcheck() -> str:
            Sends a GET request to the server to check its health status.
            Returns the health status as a string.
    """

    def __init__(self, server_url):
        """
        Initializes a ChemRecognitionClient instance.

        Parameters:
            server_url (str): The URL of the ChemICR server.
        """
        self.server_url = server_url

    def classify_image(self, image_path: str):
        """
        Sends a POST request to the server with the specified image path or directory.
        Returns the classification results in dictionary format.

        Parameters:
            image_path (str): The path to the image file or directory to be processed.

        Returns:
            dict: Classification results, including image ID, predicted labels, and chemical structures.
        """
        try:
            # Create a dictionary with the image path or directory with images
            data = {'image_path': image_path}
            # Send a POST request to the server
            response = requests.post(f'{self.server_url}/classify_image', data=data)
            response.raise_for_status()  # Raise an HTTPError for bad responses

            # Parse the JSON response
            return response.json()

        except requests.exceptions.RequestException as e:
            return {'error': str(e)}

    def healthcheck(self):
        """
        Sends a GET request to the server to check its health status.
        Returns the health status as a string.

        Returns:
            str: The health status of the ChemICR server.
        """
        try:
            # Send a GET request to the server
            response = requests.get(f'{self.server_url}/healthcheck')
            response.raise_for_status()  # Raise an HTTPError for bad responses
            # Parse the JSON response
            return response.json()

        except requests.exceptions.RequestException as e:
            return {'error': str(e)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Client for  Chemical Images Classifier and Recognizer")
    parser.add_argument("--image_path", type=str, required=True, help="Directory containing the image files.")
    parser.add_argument("--export_dir", type=str, default=".", help="Export directory for the results.")
    args = parser.parse_args()

    # URL for the combined image processing endpoint
    server_url = 'http://127.0.0.1:5000'

    # Create an instance of the client
    client = ChemClassifierClient(server_url)

    # Check the health of the server
    health_status = client.healthcheck().get('status')
    print(f"Health Status: {health_status}")

    # Check if the path to the file is valid
    file_path = Path(args.image_path)
    if not file_path.exists():
        raise FileNotFoundError(f'Invalid path or directory provided: {args.image_path}, try again with correct path')
    else:
        # Send POST request to classify and predict images
        recognition_results = client.classify_image(args.image_path)
        print(recognition_results)
        # Convert results to pands Dataframe
        df = pd.DataFrame(recognition_results)
        df.sort_values(by='image_id', inplace=True, ignore_index=True)
        print(df)
        df.to_csv(f'{args.export_dir}/{file_path.name}_{datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}.csv', index=False)
        end = time.time()
        print(f"Work took {time.strftime('%H:%M:%S', time.gmtime(end - start))}")
