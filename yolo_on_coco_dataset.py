from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # load a pretrained model

video_path = "video.mp4"

result = model(video_path)

result[0].save("bird_output.mp4")    

from ultralytics import YOLO


#now for images 


image_path = "img1.jpeg"
          
results = model(image_path)
results[0].save('output.jpeg')

