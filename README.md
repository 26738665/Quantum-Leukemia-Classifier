```markdown
# Hybrid Quantum-Classical Model for Leukemia Diagnosis

A project exploring a hybrid quantum-classical machine learning model for the classification of Acute Lymphoblastic Leukemia (ALL) from blood cell images. This repository contains the full end-to-end pipeline, from data preprocessing to the training and evaluation of both a classical baseline and a prototype quantum neural network.

**Author:** Cliff Mwendwa  
**Institution:** University of Nairobi, Chiromo Campus  
**Program:** B.Sc. in Microprocessor Technology and Instrumentation

---

## 🚀 Project Overview

The goal of this project is to investigate the potential of Quantum Machine Learning (QML) in the field of medical diagnostics. By building a hybrid model, we leverage a powerful classical Convolutional Neural Network (CNN) for feature extraction and a variational quantum circuit for classification, combining the strengths of both computational paradigms.

### Key Results
*   **Classical Baseline Accuracy:** **85.14%**
*   **Hybrid Quantum Model (v2) Accuracy:** **61.05%**

While the classical model currently outperforms the quantum prototype, this project successfully establishes a functional pipeline and provides a crucial benchmark for future work aimed at achieving a quantum advantage.

## 📁 Repository Structure

```
.
├── data/
│   ├── C-NMC_Leukemia/         # Holds the raw image data (manual download required)
│   └── leukemia_data_processed.npz  # The final, processed dataset
├── .gitignore                  # Specifies files/folders for Git to ignore
├── build_dataset.py            # Script to preprocess the raw C-NMC dataset
├── train_classical_model.py    # Script to train and evaluate the baseline CNN
├── train_hybrid_model_v2.py    # Script to train and evaluate the enhanced hybrid QML model
├── classical_leukemia_model.keras   # Saved file for the trained classical model
├── hybrid_quantum_leukemia_model_v2.keras # Saved file for the trained hybrid model
├── training_history.png        # Performance graph for the classical model
├── hybrid_training_history_v2.png # Performance graph for the hybrid model
└── requirements.txt            # All Python dependencies and their exact versions
```

## ⚙️ Setup and Installation

This project was built using a specific set of compatible libraries on **Python 3.10**. To ensure reproducibility, it is highly recommended to follow these setup steps precisely.

### 1. Prerequisite: Install Python 3.10

Ensure you have **Python 3.10.11** installed and that it is correctly added to your system's PATH. It can be downloaded from the [official Python website](https://www.python.org/downloads/release/python-31011/).

### 2. Clone the Repository

```bash
git clone https://github.com/26738665/Quantum-Leukemia-Classifier.git
cd Quantum-Leukemia-Classifier
```

### 3. Create and Activate a Virtual Environment

```bash
# Create the environment
python -m venv .venv

# Activate the environment (on Windows PowerShell)
.\.venv\Scripts\activate
```

### 4. Install Dependencies

Install all the required libraries using the pinned `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Download the Dataset

The C-NMC 2019 dataset must be downloaded manually from Kaggle.

1.  Download the data from [this Kaggle link](https://www.kaggle.com/datasets/alexkudin/c-nmc-leukemia).
2.  Create a `data` folder in the project root.
3.  Unzip the downloaded file and place the `C-NMC_Leukemia` folder inside the `data` directory. The final path should be `data/C-NMC_Leukemia/`.

## 🔬 How to Run the Pipeline

The scripts are designed to be run in a specific order.

### Step 1: Build the Dataset

This script processes the raw images and creates the final `leukemia_data_processed.npz` file. **This only needs to be run once.**

```bash
python build_dataset.py
```

### Step 2: Train the Classical Baseline Model

This script trains the CNN, evaluates it, and saves the `classical_leukemia_model.keras` file.

```bash
python train_classical_model.py
```

### Step 3: Train the Hybrid Quantum Model

This script trains the enhanced hybrid QML model (v2), evaluates it, and saves the `hybrid_quantum_leukemia_model_v2.keras` file. **Be aware: this is computationally intensive and will take a long time to run.**

```bash
python train_hybrid_model_v2.py
```

## 📈 Future Work & Improvements

The current hybrid model shows signs of overfitting and has significant room for improvement. The strategic roadmap includes:
1.  **Massive-Scale Training:** Re-training the hybrid model on the full dataset using cloud computing resources.
2.  **Advanced Quantum Circuits:** Exploring more powerful circuit designs like data re-uploading classifiers.
3.  **Sophisticated Encoding:** Using techniques like PCA or autoencoders to create more information-rich inputs for the quantum circuit.
4.  **Integration of Genomic Data:** Combining image features with genomic data to create a multi-modal diagnostic tool.
```
