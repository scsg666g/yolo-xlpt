from ultralytics import YOLO

model = YOLO("/data/dmxjcgd/yolo/models/yolo26x.pt")  # load a pretrained YOLO26x detection model
a = model("/data/dmxjcgd/yolo/ultralytics/ultralytics/assets/bus.jpg", save=True)  # predict and save the annotated image
