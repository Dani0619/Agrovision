import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ======================================
# Load Trained Model
# ======================================
print("\nLoading AI Model...")
model = load_model("plant_disease_model.keras")
print("Model loaded successfully!\n")

# ======================================
# Class Names (Same order as training)
# ======================================
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# ======================================
# Get Image Name from User
# ======================================
img_name = input("Enter image name (Example: tomato1.jpg): ")

img_path = f"test_images/{img_name}"

# ======================================
# Load Image
# ======================================
try:
    img = image.load_img(img_path, target_size=(224, 224))
except FileNotFoundError:
    print("\nImage not found!")
    print("Make sure the image is inside the 'test_images' folder.")
    exit()

# ======================================
# Preprocess Image
# ======================================
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# ======================================
# Predict
# ======================================
prediction = model.predict(img_array)

predicted_index = np.argmax(prediction)
predicted_class = class_names[predicted_index]
confidence = np.max(prediction) * 100

# ======================================
# Print Result
# ======================================
print("\n" + "="*55)
print("        AGROVISION - PLANT DISEASE DETECTION")
print("="*55)
print(f"Uploaded Image    : {img_name}")
print(f"Predicted Disease : {predicted_class}")
print(f"Confidence Score  : {confidence:.2f}%")
print("="*55)

# ======================================
# Display Image
# ======================================
plt.figure(figsize=(6,6))
plt.imshow(img)
plt.title(
    f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%",
    fontsize=12
)
plt.axis("off")
plt.show()