from ultralytics import YOLO
model = YOLO("weights/best.pt")

result = model.predict(source="0", show=True)