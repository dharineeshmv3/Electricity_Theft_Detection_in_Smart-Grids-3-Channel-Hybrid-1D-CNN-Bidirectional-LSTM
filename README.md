# Deep Learning Case Study: Electricity Theft Detection in Smart Grids
## Best Model Architecture: 3-Channel Hybrid 1D-CNN + Bidirectional LSTM (TensorFlow / Keras)

---

### Case Study Overview & Application Background
Electricity theft and Non-Technical Losses (NTL) represent over **$96 billion USD in annual losses** for electric power utilities globally. In addition to direct revenue loss, illicit power diversion causes distribution transformer overloading, unmetered phase imbalances, and severe electrical fire hazards.

With the deployment of Advanced Metering Infrastructure (AMI), smart meters record daily electricity consumption profiles across tens of thousands of households. However, detecting electricity theft using deep learning presents major real-world challenges:
1. **Extreme Class Imbalance:** Legitimate consumers vastly outnumber thieves (~8.5% theft vs. 91.5% normal, an imbalance ratio of **1 : 10.73**).
2. **High-Dimensional Sequential Data:** Each consumer has **1,034 continuous daily readings** spanning nearly 3 years (2014 to 2016), containing weekly periodicity, seasonal variations, and weather-driven demand fluctuations.
3. **Sensor Noise & Missing Data:** Meter communication dropouts (NaNs) and deliberate bypasses (0 kWh) require multi-channel spatial encoding rather than naive interpolation.

### Selected Best Deep Learning Model: 3-Channel Hybrid 1D-CNN + BiLSTM
To address both localized daily tampering patterns and long-term seasonal consumption drift, this case study implements a **3-Channel 1D Convolutional Neural Network (CNN) and Bidirectional Long Short-Term Memory (BiLSTM)** architecture using **TensorFlow / Keras** in a clean, procedural workflow without object-oriented programming (OOP) boilerplate.

---

### Table of Contents
1. Environment Setup & Library Imports
2. SGCC Dataset Acquisition & Loading
3. Exploratory Data Analysis & Class Imbalance Assessment
4. IEEE Standard Preprocessing & 3-Channel Multi-Modal Tensor Construction
5. Stratified Dataset Splitting & Class Weight Calculation
6. 3-Channel Hybrid 1D-CNN + BiLSTM Model Architecture Definition
7. Model Compilation & Cost-Sensitive Training
8. Training & Validation Performance Visualization
9. Test Set Evaluation & Operational Metric Analysis
10. Operational Threshold Tuning & Visualization (ROC, PR Curve, Confusion Matrix)
11. Individual Consumer Inference Function (`predict_consumer`)
12. Key Observations, Technical Inferences & Grid Deployment Guidelines
