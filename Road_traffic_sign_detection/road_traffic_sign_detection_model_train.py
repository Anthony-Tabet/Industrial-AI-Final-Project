from ultralytics import YOLO
import tensorflow as tf
import torch
if __name__ == '__main__':
    # Load a model
    model = YOLO("YOLOv8x.yaml")  # Specify model scale
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    model.train(
        data=r'C:\Users\Antho\OneDrive\Desktop\Industrial AI Final Project\Road_traffic_sign_detection\data\data.yaml',
        epochs=50, imgsz=640, name="trainFolderFromScratch", batch=32, workers=2)

