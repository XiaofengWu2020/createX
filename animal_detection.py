from ultralytics import YOLO

# # all from ultralytics
# # Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# # Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = model.export(format="onnx")  # export the model to ONNX format
# used pre-trained model from volo
# model = YOLO("runs/detect/train2/weights/best.pt")
# for i in range(1000):
#     results = model("africa_wild_life.jpeg", show = True)

# print(" =================================================================================  ")

#train the 10GB model but with only a few types in train images and labels, and empty val dataset
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# results = model.train(data="config.yaml", epochs = 20)

# model = YOLO("runs/detect/train20/weights/best.pt")
model = YOLO("best.pt")
results = model("several_zebras.mp4", save = True)
#results = model("africa_wild_life.jpeg", show = True, save = True)
