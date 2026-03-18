# Seismic Event Detection — Neural Binary Classifier

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-%23EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.14-3670A0?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Preprocessing-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](https://opensource.org/licenses/MIT)

---

## Overview

This project implements a **Feedforward Neural Network (MLP)** in PyTorch to perform binary classification on seismic sensor data. The objective is to determine — given two continuous underground measurements — whether a seismic event is occurring or not.

While Support Vector Machines (SVMs) are the conventional baseline for this type of dataset, this project deliberately takes a deep learning approach. The goal is not only to achieve strong classification performance, but to demonstrate how a neural network can learn non-linear decision boundaries from low-dimensional geophysical features — something classical linear models struggle with when the underlying data distribution is inherently circular or curved.

---

## Dataset

**File:** `seismic_activity_svm.csv`

| Property | Value |
|---|---|
| Total Samples | 400 |
| Features | 2 (continuous) |
| Target | Binary (0 / 1) |
| Class Balance | Perfectly balanced — 200 negative, 200 positive |
| Missing Values | None |

### Feature Descriptions

**`underground_wave_energy`** — A continuous measurement representing the magnitude of subsurface seismic wave energy. Values range from approximately −10.0 to +10.0.

**`vibration_axis_variation`** — A continuous measurement representing the directional shift or deviation along the vibration axis. Also ranging from approximately −10.0 to +10.0.

**`seismic_event_detected`** — The binary target label. `1` indicates a confirmed seismic event; `0` indicates normal baseline activity.

The perfectly balanced class distribution eliminates the need for class-weighting or oversampling strategies, making training straightforward and metrics directly interpretable.

---

## Model Architecture

The classifier is a custom `SimpleModel` class inheriting from `torch.nn.Module`. It implements a three-layer fully connected network that progressively expands the feature space before collapsing to a single output logit.

```
Input (2)  →  Linear(2 → 5)  →  ReLU  →  Linear(5 → 7)  →  ReLU  →  Linear(7 → 1)  →  Sigmoid*
```

> *Sigmoid is applied externally at inference time via `torch.sigmoid()`. During training, raw logits are passed directly to `BCEWithLogitsLoss` for numerical stability.

### Layer-by-Layer Breakdown

| Layer | Type | Input Dim | Output Dim | Activation |
|---|---|---|---|---|
| `layer_1` | `nn.Linear` | 2 | 5 | ReLU |
| `layer_2` | `nn.Linear` | 5 | 7 | ReLU |
| `layer_3` | `nn.Linear` | 7 | 1 | — (logit) |

The expansion from 2 → 5 → 7 dimensions before the final compression is an intentional design choice. By temporarily projecting the data into a higher-dimensional space, the hidden layers gain the representational capacity to carve out non-linear decision boundaries that a single linear layer could never produce.

**Why ReLU?** The Rectified Linear Unit `f(x) = max(0, x)` serves two purposes: it introduces the non-linearity the network needs to model complex patterns, and it avoids the vanishing gradient problem that plagues `tanh` and `sigmoid` activations in intermediate layers — making weight updates during backpropagation more stable and efficient.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Loss Function | `BCEWithLogitsLoss` |
| Optimizer | `Adam` |
| Learning Rate | `0.01` |
| Epochs | `200` |
| Train / Test Split | `80% / 20%` (320 / 80 samples) |
| Random Seed | `42` |
| Batch Mode | Full-batch gradient descent |

**Loss Function:** `BCEWithLogitsLoss` combines a `Sigmoid` layer and Binary Cross-Entropy loss into a single numerically stable operation. This is strongly preferred over applying `Sigmoid` first and then using `BCELoss`, which can produce `NaN` gradients at extreme logit values.

**Optimizer:** Adam (Adaptive Moment Estimation) adapts the learning rate for each parameter individually using estimates of first and second moment gradients. This makes it significantly more robust than vanilla SGD, especially on low-data regimes.

---

## Evaluation

At every epoch, both training and test set performance are tracked simultaneously. Every 10 epochs, the following metrics are printed:

```
Epoch: 0  | Loss: 0.7421 | Acc: 50.00% | Test_Loss: 0.7198 | Test_Acc: 52.50%
Epoch: 10 | Loss: 0.6712 | Acc: 56.25% | Test_Loss: 0.6589 | Test_Acc: 58.75%
...
```

The `calculate_accuracy` function computes accuracy by comparing rounded sigmoid outputs against ground truth labels:

```python
correct = torch.eq(y_true, y_pred).sum().item()
accuracy = (correct / len(y_pred)) * 100
```

---

## Visualizations

Training history is visualized using a side-by-side `seaborn` / `matplotlib` subplot generated by `plot_training_history()`:

- **Left — Training Loss** (crimson): The BCE loss curve, which should decrease monotonically toward zero as the model converges.
- **Right — Training Accuracy** (seagreen): Classification accuracy on the training set across all 200 epochs.

These curves together reveal whether the model is underfitting (both metrics plateau early), overfitting (training accuracy continues rising while test accuracy stagnates), or converging healthily.

---

## Project Structure

```
seismicevent-classifier/
│
├── main.ipynb                  # Full training pipeline & visualizations
├── seismic_activity_svm.csv    # Dataset (400 samples, 3 columns)
└── README.md
```

---

## Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/abdulkadiripek/seismicevent-classifier.git
cd seismicevent-classifier
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install torch pandas matplotlib seaborn scikit-learn
```

**4. Launch the notebook**
```bash
jupyter notebook main.ipynb
```

Execute the cells sequentially to load data, initialize the model, run the training loop, and produce all performance visualizations.


---

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
