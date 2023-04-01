from ultralytics import YOLO
import cv2 as cv
from collections import Counter

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
        results = model.predict(source="example.png", show=True)
        # for result in results:
        #     print(result.boxes.conf.numpy())
        classes = results[0].boxes.cls.numpy()
        currCounter = Counter(classes)

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
        prevCounter = currCounter

        print(prevCounter)

    # If captured image is corrupted, moving to else part
    else:
        print("No image detected. Please! try again")

