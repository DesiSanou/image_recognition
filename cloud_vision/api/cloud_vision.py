import logging
from base64 import b64encode

import googleapiclient.discovery
from oauth2client.client import GoogleCredentials


CREDENTIALS_FILE = "credentials.json"

# TODO: Make it possible to send a single request for multiple image.

class CloudVisionAPI:
    def __init__(self, credential_file=CREDENTIALS_FILE):
        # Connect to the Google Cloud-ML Service
        self.credential_file = credential_file
        self.credentials = GoogleCredentials.from_stream(credential_file)
        self.service = googleapiclient.discovery.build('vision', 'v1', credentials=self.credentials)
        self.feature_type='LABEL_DETECTION'

    def send_request(self, image, feature_type='LABEL_DETECTION'):
        """
        Send the request to Google

        :param image: image to decode
        :param feature_type: type of feature to detect 'LABEL_DETECTION' or 'TEXT_DETECTION'
        :param display: True to display the response

        :return: response from cloud vision
        """
        encoded_image_data = self._convert_image(image)
        request = self._create_request(encoded_image_data,feature_type )
        response = request.execute()
        self.feature_type = feature_type
        print(response)
        return response

    def get_content(self, response, display=False):

        # Check for errors

        if 'error' in response:
            response_content = response["error"]
            logging.error(response_content)
        else:
            if self.feature_type == 'LABEL_DETECTION':
                response_content = response['responses'][0]['labelAnnotations']

            elif self.feature_type == 'TEXT_DETECTION':
                response_content = response['responses'][0]['textAnnotations'][0]["description"]

            if display:
                self._display(response_content, feature_type)
        return response_content

    @staticmethod
    def _convert_image(image):
        """Read file and convert it to a base64 encoding"""
        with open(image, "rb") as f:
            image_data = f.read()
            encoded_image_data = b64encode(image_data).decode('UTF-8')
            return encoded_image_data

    def _create_request(self, encoded_image_data, feature_type='LABEL_DETECTION'):
        """Create the request object for the Google Vision API"""
        batch_request = [{
            'image': {
                'content': encoded_image_data
            },
            'features': [
                {
                    'type': feature_type
                }
            ]
        }]
        request = self.service.images().annotate(body={'requests': batch_request})
        return request

    @staticmethod
    def _display(response, feature_type):
        """Print the results"""
        if feature_type == 'LABEL_DETECTION':
            for label in response:
                print(label['description'], label['score'])
        elif feature_type == 'TEXT_DETECTION':
            # Print the first piece of text found in the image
            extracted_text = response[0]
            print(extracted_text['description'])

            # Print the location where the text was detected
            print(extracted_text['boundingPoly'])

