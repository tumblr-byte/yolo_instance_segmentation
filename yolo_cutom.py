#pip install ultralytics 

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-seg.yaml")  # build a new model from YAML
model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolov8n-seg.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="/content/drive/MyDrive/Butterfly/dataset.yaml", epochs=100, imgsz=640)



#test the model  for images

# Load a model
model = YOLO("yolov8n-seg.pt")  # load an official model
model = YOLO("/content/runs/segment/train/weights/best.pt")  # load a custom model

# Predict with the model
results = model("/content/img5.jpeg")

#test the model for videos

model = YOLO("/content/runs/segment/train/weights/best.pt")  # Load your custom model

# Run YOLOv8 on the video and save the results
results = model.predict(source="/content/video.mp4", save=True, conf=0.5 , show = True)

# Save the processed video path (YOLO saves it automatically in the runs directory)
output_video_path = results[0].path  # The output path for the processed video




