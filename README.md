# 🔒 Threat Detection & Prioritization

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Issues](https://img.shields.io/github/issues/vashishth-182/Threat-Detection-Prioritization)](https://github.com/vashishth-182/Threat-Detection-Prioritization/issues)
[![Star](https://img.shields.io/github/stars/vashishth-182/Threat-Detection-Prioritization?style=social)](https://github.com/vashishth-182/Threat-Detection-Prioritization/stargazers)

> 🚀 An AI-powered system to *detect, prioritize, and test cyber threats* efficiently.  
> Designed to reduce alert fatigue and highlight the most critical threats first.

---

## 📌 Overview

This project provides a *Python-based machine learning pipeline* for intrusion and threat detection with prioritization.  
Key objectives:

- ✅ Train ML models on datasets (e.g. *KDD Cup 99*) for intrusion detection
- ✅ Generate *mini test datasets* for faster experimentation
- ✅ *Prioritize* alerts by severity & likelihood
- ✅ Provide a *simple CLI app* (app.py) for testing and evaluation

---

## 📂 Repository Structure

bash
Threat-Detection-Prioritization/
│── app.py              # CLI interface for real-time/batch threat detection & prioritization
│── train_model.py      # Script to train the ML detection model
│── make_mini_test.py   # Utility to create smaller test datasets
│── mini_test.txt       # Example mini test dataset
│── KDDTest+.txt        # Example full dataset (KDD-style)
│── requirements.txt    # Python dependencies
│── README.md           # Project documentation


---

## ✨ Features

- 🔍 *Threat Detection* – Train and test ML models on security datasets
- ⚡ *Mini Test Generator* – Generate smaller subsets for faster evaluation
- 📊 *Threat Prioritization* – Rank alerts by severity & probability
- 🖥 *CLI Interface* – Run detections directly using app.py
- 📈 Extendable for *SOC (Security Operations Center)* environments

---

## ⚙ Setup & Installation

1. *Clone the repository*

   bash
   git clone https://github.com/vashishth-182/Threat-Detection-Prioritization.git
   cd Threat-Detection-Prioritization
   

2. *Install dependencies*
   bash
   pip install -r requirements.txt
   

---

## 🚀 Usage

### 🔹 Train the model

bash
python train_model.py


### 🔹 Generate a mini test dataset

bash
python make_mini_test.py


### 🔹 Run the app (threat detection + prioritization)

bash
python app.py


➡ Use mini_test.txt or KDDTest+.txt as input datasets to evaluate performance.

---

## 🧪 Data & Testing

- 📂 KDDTest+.txt → Full dataset (KDD Cup style)
- 📂 mini_test.txt → Lightweight dataset for quick testing
- ⚡ Use make_mini_test.py to create *custom subsets* for experiments

---

## 📦 Requirements

- Python *3.x*
- Core libraries:
  - scikit-learn
  - numpy
  - pandas
  - (see requirements.txt for complete list)

⚡ Training on the full dataset may take *minutes to hours* depending on hardware.

---

## 🤝 Contributing

Contributions are welcome!

👉 Fork the repo, create a branch, and open a PR.

---

## 📬 Contact

👤 *Vashishth* - [@vashishth-182](https://github.com/vashishth-182) | 👤 *Jaimil* - [@JaimilModi](https://github.com/JaimilModi)  | 👤 *Hiren* - [@Hiren-Sarvaiya](https://github.com/Hiren-Sarvaiya)

⭐ If you found this useful, don’t forget to *star the repo* → [Threat-Detection-Prioritization](https://github.com/vashishth-182/Threat-Detection-Prioritization)