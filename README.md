[README.md](https://github.com/user-attachments/files/32567505/README.md)
<div align="center">

# 🧠 Real-Time Handwritten Digit Recognizer
### *Deep Learning CNN with Interactive Pygame Canvas*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](#)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](#)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](#)
[![Pygame](https://img.shields.io/badge/Pygame-Interactive%20UI-yellowgreen?style=for-the-badge&logo=gamecontroller&logoColor=white)](#)

<br/>

<p align="center">
  <b>Draw digits in real-time and watch a Convolutional Neural Network classify them on the fly!</b>
  <br />
  <sub>Trained on the classic MNIST dataset with custom computer-vision preprocessing for high-accuracy interactive inference.</sub>
</p>

---

</div>

## 📌 Table of Contents
- [✨ Key Features](#-key-features)
- [🖥️ Interactive Demo & Controls](#️-interactive-demo--controls)
- [🏗️ System Architecture & Workflow](#️-system-architecture--workflow)
- [🧠 CNN Model Architecture](#-cnn-model-architecture)
- [📂 Project Structure](#-project-structure)
- [🚀 Quick Start & Installation](#-quick-start--installation)
- [📊 Model Training & Notebook](#-model-training--notebook)
- [💡 Tips for Best Prediction Accuracy](#-tips-for-best-prediction-accuracy)

---

## ✨ Key Features

- **⚡ Real-Time Interactive Canvas:** Draw digits freely using a responsive Pygame drawing board.
- **🎯 Dynamic Boundary Detection:** Auto-crops the drawn digit by tracking mouse coordinates and adding a boundary margin.
- **🔄 Smart MNIST Preprocessing:** Centers and pads the drawn digit (`cv2.resize` + `np.pad`) to accurately match the original MNIST distribution (`28×28` grayscale).
- **🧠 Robust CNN Backbone:** Multi-layer Convolutional Neural Network trained with Dropout regularization to prevent overfitting.
- **🛡️ Overfitting Prevention:** Early stopping and model checkpointing implemented during training to preserve the optimal weights (`best_model.h5`).
- **⌨️ Instant Canvas Reset:** Clear the screen instantly with a single keystroke (`N`).

---

## 🖥️ Interactive Demo & Controls

```
+--------------------------------------------------------------+
| Pygame Window (640 x 480)                       [ - | □ | X ]|
+--------------------------------------------------------------+
|                                                              |
|                     * * *                                    |
|                   *       *                                  |
|                         *                                    |
|                       *                                      |
|                     *                                        |
|                   * * * * *                                  |
|                   [ Two ] <-- Real-time Classification       |
|                                                              |
|                                                              |
| Controls: [Draw: Left Click] | [Reset: 'N'] | [Exit: ESC/X]  |
+--------------------------------------------------------------+
```

### 🎮 Keyboard & Mouse Shortcuts

| Action | Input | Description |
| :--- | :--- | :--- |
| **Draw Digit** | `Left Click + Drag` | Draws on the black canvas using smooth white brush strokes |
| **Classify Digit** | `Mouse Release` | Computes bounding box, standardizes shape, and displays predicted label |
| **Clear Canvas** | `N` Key | Resets the screen to black for your next drawing |
| **Quit App** | `Window Close (X)` | Safely terminates Pygame and exits the script |

---

## 🏗️ System Architecture & Workflow

```text
┌─────────────────────────────────┐
│    Interactive Pygame Canvas    │ ◄──────────────────────────┐
│      (User draws 0-9 digit)     │                            │
└────────────────┬────────────────┘                            │
                 │  Mouse Release Event                        │
                 ▼                                             │
┌─────────────────────────────────┐                            │
│      ROI Bounding Box Crop      │                            │
│  (Min/Max coordinates + margin) │                            │
└────────────────┬────────────────┘                            │
                 │  pygame.PixelArray                          │
                 ▼                                             │
┌─────────────────────────────────┐                            │
│    OpenCV Preprocessing (CV2)   │                            │
│  • Resize & 10px Center-Padding │                            │ [HUD Output]
│  • Normalize pixels (0.0 - 1.0) │                            │ Render label on canvas
└────────────────┬────────────────┘                            │ (e.g., "Seven")
                 │  Input Tensor (1, 28, 28, 1)                │
                 ▼                                             │
┌─────────────────────────────────┐                            │
│      Keras Trained CNN Model    │                            │
│  • Conv2D ➔ MaxPool ➔ Dropout   │                            │
│  • Dense Softmax Classification │                            │
└────────────────┬────────────────┘                            │
                 │  Softmax Probability Distribution           │
                 ▼                                             │
┌─────────────────────────────────┐                            │
│     Argmax Class Prediction     │ ───────────────────────────┘
│    Maps index to "Zero"–"Nine"  │
└─────────────────────────────────┘
```

1. **User Interaction:** The user draws any digit from `0` to `9` on the Pygame interface.
2. **ROI Extraction:** On mouse release, the minimum and maximum X/Y coordinates determine the bounding box.
3. **Preprocessing:**
   - Transpose and slice the active pixels from `pygame.PixelArray`.
   - Resize to intermediate scale and apply zero-padding (`10px`) to replicate MNIST digit centering.
   - Resize to standard `28×28×1` and normalize pixel values from `[0, 255]` to `[0.0, 1.0]`.
4. **Inference:** The pretrained CNN runs inference, generating class probabilities across 10 output classes.
5. **HUD Feedback:** The predicted string label (e.g., `Eight`) is rendered directly over the canvas adjacent to the drawn figure.

---

## 🧠 CNN Model Architecture

The deep learning model is built using TensorFlow/Keras and trained on the **MNIST** dataset (60,000 train / 10,000 test samples).

```
Layer (type)                 Output Shape              Param #   Activation
===========================================================================
Conv2D (32 filters, 3x3)     (None, 26, 26, 32)        320       ReLU
MaxPooling2D (2x2)           (None, 13, 13, 32)        0         -
Conv2D (64 filters, 3x3)     (None, 11, 11, 64)        18,496    ReLU
MaxPooling2D (2x2)           (None, 5, 5, 64)          0         -
Flatten                      (None, 1600)              0         -
Dropout (rate=0.25)          (None, 1600)              0         -
Dense (10 units)             (None, 10)                16,010    Softmax
===========================================================================
Total params: 34,826 (Trainable: 34,826)
```

- **Loss Function:** `CategoricalCrossentropy`
- **Optimizer:** `Adam`
- **Regularization:** `Dropout(0.25)` & `EarlyStopping(patience=5, min_delta=0.01)`
- **Checkpointing:** Saves top-performing checkpoint to `best_model.h5` based on `val_accuracy`.

---

## 📂 Project Structure

```bash
📦 handWrittenDigitRecognition
 ┣ 📜 app.py               # Real-time Pygame GUI & inference application
 ┣ 📜 j.ipynb              # Jupyter Notebook: Data loading, EDA, CNN training & evaluation
 ┣ 📜 best_model.h5        # Serialized best CNN model weights
 ┣ 📜 .gitignore           # Git ignore file
 ┗ 📜 README.md            # Project documentation & setup guide
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/handWrittenDigitRecognition.git
cd handWrittenDigitRecognition
```

### 2. Set Up a Virtual Environment *(Recommended)*
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install pygame tensorflow opencv-python numpy matplotlib
```

*(Or create a `requirements.txt` containing the above packages and run `pip install -r requirements.txt`)*

### 4. Run the Real-Time Drawing App
```bash
python app.py
```

> **Note:** In [app.py](file:///c:/Users/ranid/Downloads/AI_Project/app.py#L17), ensure the model path is set to relative path `'best_model.h5'` so it runs seamlessly on any machine:
> ```python
> MODEL = load_model('best_model.h5')
> ```

---

## 📊 Model Training & Notebook

If you wish to retrain or experiment with different neural network architectures:

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook j.ipynb
   ```
2. Execute the cells to:
   - Download and visualize the MNIST dataset.
   - Preprocess training and testing images.
   - Train the CNN with Adam optimizer and validation callbacks.
   - Evaluate loss and accuracy on unseen test data.
   - Automatically export weights to `best_model.h5`.

---

## 💡 Tips for Best Prediction Accuracy

- **Draw centrally:** Draw your numbers reasonably large and near the center of your strokes.
- **Single continuous strokes:** MNIST digits are continuous; try to draw digits without large gaps.
- **Clear between drawings:** Press **`N`** before starting a new digit to avoid bounding-box overlap.

---


<div align="center">
  <sub>Built with ❤️ using Python, Pygame, OpenCV & TensorFlow</sub>
</div>
