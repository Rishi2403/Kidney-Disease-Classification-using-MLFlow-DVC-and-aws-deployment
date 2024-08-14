import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        
        # Load the model
        model_path = os.path.join("model", "model.h5")
        model = load_model(model_path)

        # Load and preprocess the image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Make a prediction
        predictions = model.predict(test_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]

        # Define the class labels
        class_labels = {0: 'Cyst', 1: 'Normal', 2: 'Stone', 3: 'Tumor'}

        # Retrieve the predicted class label
        prediction = class_labels.get(predicted_class_index, "Unknown")

        print(f"Predicted class index: {predicted_class_index}")
        print(f"Prediction: {prediction}")

        return [{"image": prediction}]