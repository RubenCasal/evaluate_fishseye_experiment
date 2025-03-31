from ultralytics import YOLO

def main():
    # Cargar el modelo preentrenado YOLOv9m
    model = YOLO("yolov9m.pt")

    model.model.names = ['person']

    # Configuración del entrenamiento
    model.train(
        data="./fisheye_person_dataset/data.yaml",  # Ruta al archivo data.yaml
        epochs=20,                # Número de épocas
        imgsz=640,                # Tamaño de la imagen
        batch=4,                 # Tamaño del batch
        device=0,                 # Selección de GPU
        project="fine_tuning_yolov9m",  # Carpeta de resultados
        name="finetune_frozen",   # Nombre de la carpeta de ejecución
        workers=4,                # Número de trabajadores
        pretrained=True,          # Usar pesos preentrenados
        freeze=5                  # Congelar las primeras 5 capas
    )

    print("Fine-tuning completado!")

if __name__ == "__main__":
    main()
