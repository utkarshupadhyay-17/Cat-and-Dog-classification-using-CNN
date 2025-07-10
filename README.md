# Cat-and-Dog-classification-using-CNN
# 🐱🐶 Cat vs. Dog Image Classification using CNN

This project is a deep learning-based image classifier that can accurately differentiate between images of **cats** and **dogs** using a Convolutional Neural Network (CNN) model built with **TensorFlow** and **Keras**. It also includes features like **Grad-CAM visualization**, **confidence-based filtering**, and an optional **Gradio dashboard** for testing via UI.

---

## 🚀 Features

- 🧠 CNN model trained on Kaggle's Dogs vs. Cats dataset  
- 📁 Dataset organized into `cats/` and `dogs/` folders  
- 🔄 Image normalization & optional data augmentation  
- 📊 Accuracy and loss visualization  
- ❓ Rejects uncertain predictions (e.g. wrong input) using threshold  
- 🖼️ Grad-CAM heatmap to visualize model attention  
- 🖥️ Simple web dashboard using [Gradio](https://gradio.app/) (optional)

---

## 📂 Dataset

- Source: [Dogs vs. Cats Dataset on Kaggle](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
- Total images: 25,000
- Classes: `cat`, `dog`
- All images were resized to **256x256 pixels**.

> ✅ Note: You must accept Kaggle’s terms to download this dataset.

---

## 🏗️ Project Structure

├── cats-vs-dogs.ipynb # Main Colab notebook
├── sorted/
│ └── train/
│ ├── cats/
│ └── dogs/
├── model/
│ └── cat_dog_model.h5 # Saved model
├── samples/
│ └── cat_sample.jpg
│ └── dog_sample.jpg
├── README.md
└── requirements.txt



---

## 🧠 CNN Architecture

- 3 × Conv2D + ReLU + MaxPooling layers  
- Flatten → Dense(128) → Dropout  
- Dense(64) → Dropout  
- Dense(1) + Sigmoid for binary classification  
- Optimizer: `Adam`  
- Loss: `Binary Crossentropy`

---

## 🧪 Prediction Logic

```python
if pred_prob > 0.7:
    result = "Dog 🐶"
elif pred_prob < 0.3:
    result = "Cat 🐱"
else:
    result = "Unknown ❓ (low confidence)"


🖥️ Web Interface (Gradio)
1. Upload an image

2. See prediction + confidence

3. Optional download report

4. Visual explanation via Grad-CAM

5. Accuracy/Loss chart tab

Run it:

python

import gradio as gr
gr.Interface(fn=predict_image, inputs="image", outputs="label").launch()



📌 How to Use
Download and extract dataset from Kaggle.

Organize images into cats/ and dogs/ folders.

Run cats-vs-dogs.ipynb notebook to:

Train model

Evaluate performance

Test predictions

Launch Gradio interface (optional)



🙋‍♂️ Author
Utkarsh Upadhyay
B.Tech 4th year
GitHub: https://github.com/utkarshupadhyay-17

