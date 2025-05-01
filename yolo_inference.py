from ultralytics import YOLO

model = YOLO('traininig/models/last.pt')

model.predict('input_video/input_video.mp4', save=True, conf=0.3)