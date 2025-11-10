# 🩺 Breast Cancer Prediction Model

A Machine Learning project using **Logistic Regression** to classify whether a cell is **benign** or **malignant** based on medical features from the **Breast Cancer Wisconsin Dataset**.  

This project demonstrates an end-to-end ML workflow — from data preprocessing and visualization to model training, evaluation, and saving the trained model.

---

## 📘 Overview

In this project, I built and evaluated a **Logistic Regression model** to predict breast cancer diagnosis using 30 numerical features computed from fine-needle aspirate (FNA) images of breast masses.

The main objective is to develop a simple, interpretable model that can help understand which cell characteristics indicate malignancy.

---

## 📂 Dataset

The dataset used in this project comes from Kaggle:

👉 [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

---

## 💾 Files in this Repository

| File | Description |
|------|--------------|
| `notebook/breast_cancer_prediction_model.ipynb` | Main Jupyter Notebook (code, visualizations, and explanations) |
| `notebook/breast-cancer-wisconsin-data/data.csv` | Dataset used for training and testing |
| `artifacts/` | Directory containing the saved model and scaler (optional) |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## ⚙️ Project Workflow

1. **Data Loading & Exploration:**  
   - Loaded the Breast Cancer dataset using opendatasets.  
   - Explored feature names, data shape, and class distribution.  
   - Visualized correlations and feature relationships to understand the dataset better.

2. **Data Preprocessing:**  
   - Used `train_test_split` with `stratify=y` to preserve class proportions in both training and testing sets.
   - Standardized numeric features using **StandardScaler** for better model performance.  

3. **Model Building:**  
   - Trained a **Logistic Regression** model using scikit-learn.  
   - Chosen for its simplicity, interpretability, and strong performance on binary classification tasks.

4. **Model Evaluation:**  
   - Metrics used: **Accuracy**, **Precision**, **Recall**, **F1-score**, and **Confusion Matrix**  

5. **Model Saving:**  
   - Saved the trained Logistic Regression model and the scaler using `joblib` for future deployment.

---

## 📈 Key Visualizations

- **Feature Correlation Heatmap:**  
  Identifies the most influential features contributing to cancer diagnosis.

- **Confusion Matrix:**  
  Visualizes true vs. false predictions to evaluate model accuracy and recall.

---

## 🧠 Technologies Used

| Category | Tools & Libraries |
|-----------|------------------|
| Language | Python |
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn |
| Environment | Jupyter Notebook |

---

## 📊 Model Performance

| Metric | Test dataset Value (Approx.) |
|--------|-----------------|
| Accuracy | 0.97 |
| Precision | 1.00 |
| Recall | 0.93 |
| F1-score | 0.96 |

---

## 🏁 Conclusion

- The **Logistic Regression model** accurately predicts breast cancer diagnosis with around **97% accuracy**.  
- The model performs well on both malignant and benign classes, showing balanced precision and recall.  
- Features like **mean radius**, **mean concave points**, and **mean texture** have strong influence on predictions.  

---

## 🚀 Future Improvements

- Add **cross-validation** with a Scikit-learn `Pipeline` to ensure no data leakage.  
- Visualize **ROC-AUC** and **Precision-Recall** curves for more detailed performance analysis.  
- Explore advanced models like **Random Forest**, **SVM**, or **XGBoost** for comparison.  
- Build a small **Flask or Streamlit web app** for interactive prediction.

---

## 🧩 How to Run This Repository

To run this project locally, clone the repository from GitHub and open the Jupyter Notebook file.  
Install the required Python libraries (`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`) before running the notebook.  
Download the Kaggle dataset linked above and place it in the project’s `breast-cancer-wisconsin-data/` directory.  

Once the environment is set up, launch Jupyter Notebook, open `breast_cancer_prediction_model.ipynb`, and run all cells sequentially.

---

## 🙋‍♀️ Author

**Dhruvi Mevada**  
📫 [Connect on LinkedIn](https://www.linkedin.com/in/dhruvi-mevada-3526992b0)  
💻 [View my GitHub Profile](https://github.com/dhruvimvd9)

---
