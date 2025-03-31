from ultralytics import YOLO

model = YOLO("last.pt")  

metrics = model.val(
    data="./original2_top_view_person_dataset/data.yaml",  # YAML with val images + labels
    imgsz=640,                 # image size
    conf=0.5,                 # confidence threshold
    iou=0.5,                   # IoU threshold
    split="train",              # or 'test'
    verbose=True
)

print(metrics)  # dict with mAP, precision, recall, etc.
