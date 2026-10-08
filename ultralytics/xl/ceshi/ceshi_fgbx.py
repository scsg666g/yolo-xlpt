from ultralytics import YOLO

model = YOLO("/data/yolo/yolo/ultralytics/runs/detect/反光背心_yolo26x_960/weights/best.pt") 
a = model("/data/yolo/yolo/ultralytics/xl/ceshi-images/11.jpg",
        conf=0.5,
        imgsz=960,
        save=True,
        project="/data/yolo/yolo/ultralytics/runs/detect/ceshi",
        name="cs_fgbx",
        )  
