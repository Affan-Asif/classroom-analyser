from ultralytics import YOLO

model = YOLO("yolo11n.pt")

train_results = model.train(
    data= "C:/Affan/Affan/__CODING__/__PROJECTS__/Classroom Analyser/data_custom.yaml",
    epochs=5
)