# 🤓 Policy Optimization for Financial Decision-Making

This project analyzes **LendingClub Loan Data** to compare two different machine learning approaches for **loan approval**:

- **Task 2 (Predictive Model):** A deep learning classifier trained to predict the risk of default.  
- **Task 3 (Reinforcement Learning Agent):** A deep Q-network trained to maximize financial profit.

The repository contains all the necessary code to preprocess data, train both models, and generate the final analysis and report.

---

## ⚙️ Environment Setup (Google Colab Recommended)

This project is designed to run in **Google Colab** to leverage free GPU resources for training.

### 1. Install Dependencies

You can install all required Python packages using:

```bash
!pip install -r requirements.txt
```

Or manually install them in a Colab cell:

```bash
!pip install torch tqdm pandas numpy scikit-learn joblib matplotlib seaborn
```

---

### 2. Data Setup

**Download Dataset:**  
Get the dataset from [Kaggle - LendingClub Loan Data](https://www.kaggle.com/wordsforthewise/lending-club).  
You’ll need the file:  
`accepted_2007_to_2018Q4.csv.gz`

**Google Drive Configuration:**

1. Create a folder named `shodhAI` in the root of your Google Drive.  
2. Upload the `accepted_2007_to_2018Q4.csv.gz` file into that folder.  

All scripts read and save data to:  
`/content/drive/MyDrive/shodhAI/`

---

## 🚀 How to Run the Project (Step-by-Step)

Run the following scripts **in order**.

### **Step 1: (Optional) Exploratory Data Analysis**

Performs initial analysis and justifies feature selection.

```bash
!python task_1_eda.py
```

**Output:**
- EDA plots (e.g., `eda_1_loan_status_distribution.png`) saved in your Colab environment.

---

### **Step 2: Data Preprocessing**

Mandatory step.  
Builds preprocessing pipeline, engineers reward features, and saves processed datasets.

```bash
!python eda_and_preprocessing.py
```

**Output (Saved to `/shodhAI/`):**
- `processed_data_train.npz`  
- `processed_data_test.npz`  
- `preprocessor.joblib`

---

### **Step 3: Train Model 1 — Predictive Deep Learning Classifier**

Trains a deep learning classifier (Task 2) to predict loan default risk.

```bash
!python task_2_dl_model.py
```

**Output (Saved to `/shodhAI/`):**
- `dl_model_weights.pth`

**Terminal Output:**
```
--- Test Set Performance (Task 2) ---
AUC (Area Under ROC): ...
Best F1-Score: ...
Recall (at best F1): ...
```

---

### **Step 4: Train Model 2 — RL Q-Network**

Trains a deep Q-learning agent to maximize simulated financial profit.

```bash
!python task_3_rl_model.py
```

**Output (Saved to `/shodhAI/`):**
- `rl_q_model.pth`

**Terminal Output:**
```
--- Test Set Performance (Task 3) ---
Total Simulated SCALED Profit: ...
Average Simulated SCALED Profit: ...
```

---

### **Step 5: Final Analysis and Comparison**

Compares both models, analyzing cases where their policies disagree.

```bash
!python task_4_analysis.py
```

**Terminal Output:**
```
--- Model Policy Comparison ---  (Approval rates for both)
--- Analyzing 5 Example Disagreements ---  (Core analysis for report)
```

---

## 📄 Final Report

The complete analysis and comparison are summarized in  
[`Finalprojectreport.pdf`](./Finalprojectreport.pdf).

---

## 📁 Directory Structure

```
.
├── EDA.ipynb
├── FinalPreprocessing.ipynb
├── task_2_dl_model.py
├── task_3.ipynb
├── task4.ipynb
├── requirements.txt
├── Finalprojectreport.pdf

```

---

## 🧩 Technologies Used

- **Python 3.10+**
- **PyTorch**
- **NumPy / Pandas / scikit-learn**
- **Matplotlib / Seaborn**
- **Google Colab**

---

## 🧠 Project Goal

To evaluate whether a **predictive model** or a **reinforcement learning agent** yields a better financial decision policy when approving loans — balancing risk prediction against profit optimization.

---
