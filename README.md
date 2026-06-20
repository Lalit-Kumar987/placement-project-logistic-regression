# Summary🎓💼

An end-to-end Machine Learning pipeline built to predict student placement eligibility based on academic performance (**CGPA**) and cognitive score (**IQ**). This repository includes the exploratory data analysis, the training pipeline, and the serialized model (`model.pkl`) ready for integration into a viewer or deployment environment.

---

## 📊 Project Overview

This project analyzes the factors influencing campus recruitment outcomes for 100 students. The primary insights reveal that consistent academic performance (CGPA) serves as a stronger indicator for recruitment outcomes than raw IQ scores alone. 

To bridge these insights into a usable tool, a classification model was built and saved as a pickle file (`model.pkl`) to serve real-time or batch inference requests in a viewer dashboard or API.

### Pipeline Workflow:
1. **Exploratory Data Analysis (EDA):** Analyzed feature correlations and target balance.
2. **Feature Scaling:** Applied standard scaling to align the variances of CGPA and IQ scores.
3. **Model Selection:** Trained and evaluated a classification model (e.g., Logistic Regression).
4. **Decision Boundary Mapping:** Visualized the classification threshold separating placed vs. unplaced regions.
5. **Serialization:** Exported the finalized model weights and parameters to `model.pkl`.

---

## ⚙️ How to Run the Model with your Viewer

To integrate the serialized `model.pkl` into a Python viewer application (such as a Streamlit dashboard, a FastAPI backend, or a simple command-line script), follow the steps below.

### 1. Prerequisites
Ensure you have the required packages installed in your environment:

```bash
pip install numpy pandas scikit-learn
import pickle
import numpy as np

# 1. Load the serialized model pipeline
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

print("✅ Model loaded successfully!")

# 2. Define new input data: [CGPA, IQ]
# Example: Student A (CGPA: 7.5, IQ: 120) and Student B (CGPA: 5.2, IQ: 110)
new_data = np.array([
    [7.5, 120.0],
    [5.2, 110.0]
])

# 3. Generate Predictions
# Output: 1 = Placed, 0 = Unplaced
predictions = model.predict(new_data)

for i, pred in enumerate(predictions):
    status = "Placed 🎉" if pred == 1 else "Not Placed ❌"
    print(f"Student {i+1} status: {status}")
