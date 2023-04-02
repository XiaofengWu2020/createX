from ultralytics import YOLO
import cv2 as cv
from collections import Counter
import os
import replicate
from PIL import Image
import base64
import gtts
from playsound import playsound


os.environ["REPLICATE_API_TOKEN"] = "f76c82c8e540e402e79f8ce24fa69957294fef9c"
prevCounter = Counter()
model = YOLO("weights/best.pt")

while True:
    cam_port = 0
    cam = cv.VideoCapture(cam_port)
    
    # reading the input using the camera
    success, image = cam.read()
    
    # If image will detected without any error, 
    # show result
    if success: 
        # saving image in local storage
        cv.imwrite("example.png", image)
        
        # Encoding the image file to base64 string
        results = model.predict(source="example.png", show=True)
        classes = results[0].boxes.cls.numpy()
        currCounter = Counter(classes)
        
        if len(currCounter) == 0:
            prevCounter = currCounter

            print("no objects detected")
            continue
        changeOccurred = False
        for key in currCounter:
            if key in prevCounter:
                if currCounter[key] != prevCounter[key]:
                    changeOccurred = True
                    break
            else:
                changeOccurred = True
        if changeOccurred:
            print("change occurred")
            # todo: call API and convert to audio
        # for result in results:
        #     print(result.boxes.conf.numpy())
            with open("example.png", "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            model_replicate = replicate.models.get("rmokady/clip_prefix_caption")
            version = model_replicate.versions.get("9a34a6339872a03f45236f114321fb51fc7aa8269d38ae0ce5334969981e4cd8")

            #https://replicate.com/rmokady/clip_prefix_caption/versions/9a34a6339872a03f45236f114321fb51fc7aa8269d38ae0ce5334969981e4cd8#input
            #Do this in your terminal before running! export REPLICATE_API_TOKEN=d0e77e654a18ac90f64e75e1d498a45bd6a72030
            # Creating the inputs dictionary with the URI to the image file
            inputs = {
                # Input image URI
                'image': f"data:image/png;base64,{encoded_string}",

                # Choose a model
                'model': "coco",

                # Whether to apply beam search to generate the output text
                'use_beam_search': False,
            }

        # https://replicate.com/rmokady/clip_prefix_caption/versions/9a34a6339872a03f45236f114321fb51fc7aa8269d38ae0ce5334969981e4cd8#output-schema
        # The description string
            output = version.predict(**inputs)
            print(output)

            # make request to google to get synthesis
            tts = gtts.gTTS(output)

            # save the audio file
            tts.save("description.mp3")

            # play the audio file
            playsound(os.path.dirname(__file__) + "\description.mp3")

        prevCounter = currCounter

    # If captured image is corrupted, moving to else part
    else:
        print("No image detected. Please! try again")

