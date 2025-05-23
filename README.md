# Methodology for detecting post-traumatic stress disorder and other associated mental disorders in textual content

## Overview
Post-traumatic stress disorder (PTSD) and related mental health conditions often remain undetected due to the limitations of traditional diagnostic methods, which require direct human interaction and are not well-suited for identifying issues in digital communication. This project addresses the gap by applying natural language processing (NLP) and machine learning to analyze online user-generated content and detect early signs of PTSD and co-occurring disorders.

---

## Project Structure
| Folder                  | Purpose                                                                 |
|-------------------------|-------------------------------------------------------------------------|
| **datasets/**           | Raw datasets used for training and evaluation.                          |
| **model/**              | Classification models.                                                  |
| **result_interpretation/** | Outputs for interpreting model results and visualizations.           |
| **train/**              | Script for training classification models.                              |
| **trained_models/**     | Saved models and weights for PTSD and comorbid disorder detection.      |
| **utils/**              | Utility functions for preprocessing, visualization, and inference.      |
| **main.py**             | Main script to run the UI application.                                  |

---

## Team

* **Oleksandr Mazurets** – Faculty Advisor
* **Oleksandr Ovcharuk** – Team Captain, Speaker, Researcher
* **Roman Vit** – Dataset Assembler, ML Engineer, Soft Developer
* **Veronika Kadynska** – Researcher, Dataset Assembler, ML Engineer

---

## Datasets

This project utilizes several publicly accessible datasets, all of which are stored in [`datasets/`](datasets/):

| Purpose               | Dataset / Link                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------- |
| PTSD                  | [Human Stress Prediction](https://www.kaggle.com/datasets/kreeshrajani/human-stress-prediction)                     |
| PTSD                  | [aya_ptsd](https://www.kaggle.com/datasets/abdelrahmanahmed3/aya-ptsd)                                              |
| Other Disorders       | [COMSYS-T1](https://www.kaggle.com/datasets/kajimi/comsys2023)                                                      |
| Healthy               | [Depression: Reddit Dataset (Cleaned)](https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned)     |

We sincerely appreciate the efforts of all authors and organisations who provided these datasets.

---

## Installation

1. **Create a virtual environment**

   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. **Install PyTorch**
   *Pick the command that matches your CUDA version (or CPU-only).*
   Example for CUDA 12.1:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   For CPU-only:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install project dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### 1. Graphical Application

Download trained models from [Google Drive](https://drive.google.com/drive/folders/1kQQYUKK1K6fZJV4vSKcsWMcDQJCLUW0E?usp=drive_link) and place donwloaded folders inside [`trained_models/`](trained_models/) and run command:    

```bash
python main.py
```

### 2. Training the PTSD Classifier and Multi-Disorder Classifier

```bash
cd train
python train.py
```

---

## License

*(To be finalised after internal review. Until then, all code is provided for non-commercial research and educational use.)*
