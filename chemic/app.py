"""
app.py

This module provides a Flask web application for classifying chemical images using a pre-trained ResNet-50 models.

Dependencies:
    - python>=3.11
    - flask
    - image_classifier (assuming ImageClassifier class is defined in the 'image_classifier' module)

Usage:
    Start the Flask application by running this module in a production mode. Run in the folder ChemIC:
    gunicorn -w 1 -b 127.0.0.1:5000 --timeout 3600 chemic.app:app
  - -w 1: Specifies the number of worker processes. In this case, only one worker is used.
    Adjust this value based on your server's capabilities.
  - -b 127.0.0.1:5000: Binds the application to the specified address and port. Change
    the address and port as needed.
  - --timeout 3600: Sets the maximum allowed request processing time in seconds.
    Adjust this value based on your application's needs.

Models:
    - ResNet-50: A pre-trained ResNet-50 models is used for image classification.

Endpoints:
    Send images for classification and SMILES recognition through the specified endpoint:
    '/classify' [POST]
    - Accepts image paths or directories for classification.
    - Classifies the images
    - Returns a JSON response containing the classified images data and the details as list of dictionaries:
    [{
    'image_id': 'US07314693-20080101-C00055.png',
    'predicted_label': 'single chemical structure',
    'program': 'ChemIC',
    'program_version': 'CCN(CC)c1ccc(-c2ccc(N(c3ccccc3)c3ccccc3)cc2)cc1'
    },...]

    '/healthcheck' [GET]
    - Check if server is up and running.

Author:
    Dr. Aleksei Krasnov
    a.krasnov@digital-science.com
    Date: February 26, 2024
"""

from flask import Flask, request, jsonify
from .image_classifier import ImageClassifier

app = Flask(__name__)

print(f'Web service {__name__} is ready to work...')


@app.route('/classify_image', methods=['POST'])
def classify_image():
    try:
        # Get the image path from the request
        image_path = request.form['image_path']
        image_recognizer = ImageClassifier()
        image_recognizer.results = []  # Assign an attribute as empty list for each new classification cycle
        print(f'Server received image {image_path}')
        # Classification step
        image_recognizer.send_to_classifier(image_path)
        results = image_recognizer.results
        return jsonify(results), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/healthcheck', methods=['GET'])
def healthcheck_route():
    return jsonify({'status': 'Server is up and running'})


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
