---

# 🔍 AI Fake News Detector

An AI tool that checks if news is real or fake.
It uses **three methods** — classic machine learning, modern AI, and rule-based checks — for accuracy and clear results.

**Dataset:** [Kaggle Fake News Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

🚀 **[Launch App](https://fakenewsdetectorpy-passiveaggresive.streamlit.app/)**

📺 **[Watch Demo](https://youtu.be/HgOg7ztfMWA)**

> ⚠️ On Streamlit Cloud, the app may run slower because it uses CPU. The RoBERTa model works faster with GPU.

---

## 🎯 Features

* **Hybrid System:** Combines classic ML, modern AI, and keyword rules
* **High Accuracy:** Best model gets **99% accuracy**
* **Explainable:** Shows keywords and confidence scores
* **Easy to Use:** Simple Streamlit web app

---

## 🏗️ How It Works

Text goes through **three models** at the same time.
Their results are combined to give one final prediction.

1. **Classic AI — Passive Aggressive Classifier**

   * Uses word frequency (TF-IDF)
   * Very fast, small memory use, best accuracy

2. **Modern AI — RoBERTa Transformer**

   * Pre-trained model that understands context and meaning
   * Good with tricky language and sarcasm

3. **Rule-Based — spaCy + Keywords**

   * Looks for common fake-news words like "shocking" or "secret"
   * Very fast and easy to understand

---

## 📊 Model Results (300-sample test)

| Model          | Accuracy  | F1        | Speed (GPU) |
| -------------- | --------- | --------- | ----------- |
| **Classic ML** | **99.0%** | **99.0%** | 0.01s       |
| RoBERTa        | 52.3%     | 35.6%     | 0.15s       |
| Rule-Based     | 52.0%     | 51.3%     | 0.02s       |

---

## 🛠️ Run Locally

**Requirements:** Python 3.8+, 4GB RAM (8GB better)

```bash
# 1. Get the code
git clone https://github.com/Aimandzikri/Fake_News-Detector.git
cd Fake_News-Detector

# 2. (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install packages
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Run the app
streamlit run streamlit_app.py
```

---

## ⚙️ Tech Used

* **Frontend:** Streamlit
* **ML:** Scikit-learn, PyTorch
* **NLP:** Hugging Face Transformers, spaCy, NLTK
* **Plots:** Matplotlib, Seaborn

---

Do you want me to also make a **super-short one-page version** for this so it’s even easier to scan? That could make your README faster to read.
