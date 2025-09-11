# 🔒 Threat Detection & Prioritization

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Issues](https://img.shields.io/github/issues/vashishth-182/Threat-Detection-Prioritization)](https://github.com/vashishth-182/Threat-Detection-Prioritization/issues)
[![Stars](https://img.shields.io/github/stars/vashishth-182/Threat-Detection-Prioritization?style=social)](https://github.com/vashishth-182/Threat-Detection-Prioritization/stargazers)

> 🚀 An AI-powered system to **detect, prioritize, and test cyber threats** efficiently.  
> Designed to reduce alert fatigue and highlight the most critical threats first.

---

## 📌 Overview

The project provides a **Python-based machine learning pipeline** for threat detection and alert prioritization.  
Key objectives:

- ✅ Train ML models on datasets (e.g. **KDD Cup**) for **intrusion/threat detection**
- ✅ Generate **mini test datasets** for faster experimentation
- ✅ **Prioritize** alerts by severity & likelihood
- ✅ Provide a **simple CLI app** (`app.py`) for testing and evaluation

---

## 📂 Repository Structure

```bash
Threat-Detection-Prioritization/
│── app.py              # Main interface for real-time/batch threat detection
│── train_model.py      # Script to train the detection model
│── make_mini_test.py   # Utility to create smaller test datasets
│── mini_test.txt       # Sample mini test dataset
│── KDDTest+.txt        # Full dataset (KDD-style)
│── requirements.txt    # Project dependencies
│── README.md           # Project documentation
```

---

## ✨ Features

- 🔍 **Threat Detection** – Train ML models on real datasets
- ⚡ **Mini Test Generator** – Speed up evaluation with subset datasets
- 📊 **Threat Prioritization** – Rank threats by severity & probability
- 🖥 **CLI Interface** – Interact with the model using `app.py`
- 📈 Extendable for **real-world SOC usage**

---

## ⚙️ Setup & Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/vashishth-182/Threat-Detection-Prioritization.git
   cd Threat-Detection-Prioritization
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 🔹 Train the model

```bash
python train_model.py
```

### 🔹 Generate a mini test dataset

```bash
python make_mini_test.py
```

### 🔹 Run the app (threat detection + prioritization)

```bash
python app.py
```

➡️ Use `mini_test.txt` or `KDDTest+.txt` to evaluate performance.

---

## 🧪 Data & Testing

- 📂 `KDDTest+.txt` → Full dataset (KDD Cup style)
- 📂 `mini_test.txt` → Lightweight dataset for quick testing
- ⚡ Use `make_mini_test.py` to create **custom subsets**

---

## 📦 Requirements

- Python **3.x**
- Core libraries:
  - `scikit-learn`
  - `numpy`
  - `pandas`
  - (see `requirements.txt` for full list)

⚡ Model training on the full dataset may take **minutes to hours** depending on system resources.

---

## 🤝 Contributing

Contributions are welcome!  
You can help by:

- 📝 Documenting the **prioritization logic**
- 🎯 Improving ML model performance (new features / algorithms)
- 🧪 Adding **unit tests + CI/CD**
- 🌐 Building a **web dashboard / REST API**

👉 Fork the repo, create a branch, and submit a PR.

---

## 📜 License

This project is licensed under the **MIT License**.  
See [LICENSE](LICENSE) for details.

---

## 📬 Contact

👤 **Vashishth**  
🔗 GitHub: [@vashishth-182](https://github.com/vashishth-182)

👤 **Jaimil**  
🔗 GitHub: [@JaimilModi](https://github.com/JaimilModi)

👤 **Hiren**  
🔗 GitHub: [@Hiren-Sarvaiya](https://github.com/Hiren-Sarvaiya)

⭐ If you found this useful, don’t forget to **star the repo** → [Threat-Detection-Prioritization](https://github.com/vashishth-182/Threat-Detection-Prioritization)
