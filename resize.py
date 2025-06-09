import os
from PIL import Image
from torchvision import transforms

# Transformación igual que la que usarías en entrenamiento
resize_transform = transforms.Resize((260, 260))

def resize_with_transforms(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                in_path = os.path.join(root, file)
                rel_path = os.path.relpath(in_path, input_dir)
                out_path = os.path.join(output_dir, rel_path)

                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                try:
                    img = Image.open(in_path).convert("RGB")
                    img = resize_transform(img)  # <- usa transforms.Resize
                    img.save(out_path)
                except Exception as e:
                    print(f"Error procesando {in_path}: {e}")

    print("Transformación completada. Imágenes guardadas en:", output_dir)

if __name__ == "__main__":
    resize_with_transforms("Cam4", "/repo-fast/Cam4_resized_260_260")
