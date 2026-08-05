import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from disease_info import disease_info

# ============================================
# AGROVISION - Plant Disease Detection System
# ============================================

print("\n==============================================")
print("      AGROVISION AI MODEL LOADING...")
print("==============================================")

# Load Trained Model
model = load_model("plant_disease_model.keras")

print("Model Loaded Successfully!")
print("==============================================\n")

# ============================================
# Disease Classes (Same Order as Training)
# ============================================

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

# ============================================
# Ask User for Image
# ============================================

img_name = input("Enter Image Name (Example: tomato1.jpg): ")

img_path = f"test_images/{img_name}"

# ============================================
# Load Image
# ============================================

try:
    img = image.load_img(img_path, target_size=(224,224))
except FileNotFoundError:
    print("\n Image not found!")
    print("Please check the filename inside the test_images folder.")
    exit()

# ============================================
# Image Preprocessing
# ============================================

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# ============================================
# Prediction
# ============================================

prediction = model.predict(img_array)

predicted_index = np.argmax(prediction)
predicted_class = class_names[predicted_index]
confidence = np.max(prediction) * 100

# ============================================
# Fetch Disease Information
# ============================================

info = disease_info.get(predicted_class)

# ============================================
# Display Result
# ============================================

print("\n")
print("="*60)
print("        AGROVISION - PLANT DISEASE DETECTION")
print("="*60)

print(f"\nUploaded Image     : {img_name}")
print(f"Predicted Disease  : {predicted_class}")
print(f"Confidence Score   : {confidence:.2f}%")

# Low confidence warning
if confidence < 60:
    print("\n⚠ WARNING")
    print("The model confidence is low.")
    print("Please upload a clearer leaf image for better accuracy.")

# ============================================
# Disease Suggestions
# ============================================

if info:

    print("\nCause")
    print("------------------------------------------")
    print(info["cause"])

    print("\nSymptoms")
    print("------------------------------------------")
    for symptom in info["symptoms"]:
        print("•", symptom)

    print("\nTreatment")
    print("------------------------------------------")
    for treatment in info["treatment"]:
        print("•", treatment)

    print("\nPrevention")
    print("------------------------------------------")
    for prevention in info["prevention"]:
        print("•", prevention)

else:

    print("\nDisease information is not available.")
    print("Please update disease_info.py")

print("\n" + "="*60)

# ============================================
# Display Uploaded Image
# ============================================

plt.figure(figsize=(6,6))

plt.imshow(img)

plt.title(
    f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%",
    fontsize=12
)

plt.axis("off")

plt.show()