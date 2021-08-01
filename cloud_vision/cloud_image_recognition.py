import logging
from base64 import b64encode

import googleapiclient.discovery
from oauth2client.client import GoogleCredentials

# Settings
IMAGE_FILE = "road_sign.jpg"
CREDENTIALS_FILE = "credentials.json"


class CloudVisionAPI:
    def __init__(self, credential_file):
        # Connect to the Google Cloud-ML Service
        self.credential_file = credential_file
        self.credentials = GoogleCredentials.from_stream(credential_file)
        self.service = googleapiclient.discovery.build('vision', 'v1', credentials=self.credentials)

    def send_request(self, image, feature_type='LABEL_DETECTION', display=False):
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
        # Check for errors
        valid = 'error' not in response

        if feature_type == 'LABEL_DETECTION':
            response_content = response['responses'][0]['labelAnnotations']

        elif feature_type == 'TEXT_DETECTION':
            response_content = response['responses'][0]['textAnnotations']

        if display and valid:
            self._display(response_content, feature_type)

        return response, valid

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


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)

    print("\n### Image content recognition:%s\n" % (IMAGE_FILE))
    print("<pre>")
    cloud_vision = CloudVisionAPI(credential_file=CREDENTIALS_FILE)
    response, valid = cloud_vision.send_request(image=IMAGE_FILE, feature_type='LABEL_DETECTION', display=True)
    logging.debug(response)
    print("</pre>")
    print("**valid**: %r "%(valid))
    TEXT_IMAGE = "text.png"
    print("\n### Image content recognition:%s " % (TEXT_IMAGE))
    print("<pre>")
    response, valid = cloud_vision.send_request(image=TEXT_IMAGE, feature_type='TEXT_DETECTION', display=True)
    print("</pre>")
    logging.debug(response)
    print("**valid**: %r " % (valid))
