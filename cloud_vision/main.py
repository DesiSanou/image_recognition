import logging
from  api.cloud_vision import CloudVisionAPI

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)

    IMAGE_FILE = "road_sign.jpg"

    print("\n### Image content recognition:%s\n" % (IMAGE_FILE))
    print("<pre>")
    cloud_vision = CloudVisionAPI(credential_file="./api/credentials.json")
    response, valid = cloud_vision.send_request(image=IMAGE_FILE, feature_type='LABEL_DETECTION', display=True)
    logging.debug(response)
    print("</pre>")
    print("**valid**: %r "%(valid))

    TEXT_IMAGE = "text_image.png"
    print("\n### Image content recognition:%s " % (TEXT_IMAGE))
    print("<pre>")
    response, valid = cloud_vision.send_request(image=TEXT_IMAGE, feature_type='TEXT_DETECTION', display=True)
    print("</pre>")
    logging.debug(response)
    print("**valid**: %r " % (valid))
