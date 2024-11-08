from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Load your YOLO model (update path to your ONNX model)
model = YOLO("C:\\Users\\Jiggs Dev\\Desktop\\project\\train17\\weights\\best.onnx")

# Class labels
names = [
    'cat-Abyssinian', 'cat-Bengal', 'cat-Birman', 'cat-Bombay',
    'cat-British_Shorthair', 'cat-Egyptian_Mau', 'cat-Maine_Coon', 
    'cat-Persian', 'cat-Ragdoll', 'cat-Russian_Blue', 'cat-Siamese', 
    'cat-Sphynx', 'dog-american_bulldog', 'dog-american_pit_bull_terrier',
    'dog-basset_hound', 'dog-beagle', 'dog-boxer', 'dog-chihuahua', 
    'dog-english_cocker_spaniel', 'dog-english_setter', 'dog-german_shorthaired', 
    'dog-great_pyrenees', 'dog-havanese', 'dog-japanese_chin', 'dog-keeshond', 
    'dog-leonberger', 'dog-miniature_pinscher', 'dog-newfoundland', 'dog-pomeranian', 
    'dog-pug', 'dog-saint_bernard', 'dog-samoyed', 'dog-scottish_terrier', 
    'dog-shiba_inu', 'dog-staffordshire_bull_terrier', 'dog-wheaten_terrier', 
    'dog-yorkshire_terrier'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        image_file = request.files['image']
        image = Image.open(image_file.stream)
        image = image.resize((640,640))

        # Predict with YOLO
        results = model.predict(source=image)

        # Process prediction results
        predictions = []
        for result in results:
            for box in result.boxes:
                # Get box details
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                class_id = int(box.cls[0])

                # Ensure class_id is within range
                if 0 <= class_id < len(names):
                    class_name = names[class_id]
                    predictions.append({
                        "class": class_name,
                        "confidence": float(confidence),
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                        "class_id": class_id
                    })
                else:
                    return jsonify({"error": "Class ID out of range"}), 500

        return jsonify({"predictions": predictions})

    except Exception as e:
        print(f"Error processing image or running model: {str(e)}")
        return jsonify({"error": "Error processing image or running model"}), 500

if __name__ == '__main__':
    app.run(debug=True)
