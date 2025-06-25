# Textective - AI vs Human Text Detection System

A machine learning web application that classifies text as AI-generated or human-written using Support Vector Machine (SVM), Decision Tree, and AdaBoost algorithms. This project was developed for the Large Language Models course to create a comprehensive text classification system that distinguishes between AI-generated and human-written content using multiple machine learning approaches.

---

## 🚀 Features

- **Three ML Models**: SVM, Decision Tree, and AdaBoost classifiers
- **File Support**: PDF, DOCX, and TXT file uploads
- **Real-time Analysis**: Instant predictions with confidence scores
- **Interactive Visualizations**: Charts, word clouds, and model comparisons
- **Detailed Reports**: Downloadable analysis reports
- **User-friendly Interface**: Clean, responsive Streamlit web app

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Streamlit

# Clone repository
git clone https://github.com/yourusername/ai-human-detection-project.git
cd ai-human-detection-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

---

## 🏃‍♂️ Usage

### Running the app
streamlit run app.py

---

## 📁 Project Structure
ai_human_detection_project/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation
├── models/                         # Trained model files
│   ├── svm_model.pkl
│   ├── decision_tree_model.pkl
│   ├── adaboost_model.pkl
│   └── tfidf_vectorizer.pkl
├── data/                          # Training and test data
│   ├── training_data/
│   └── test_data/
└── notebooks/                     # Jupyter development notebooks
    ├── 01_data_exploration.ipynb
    ├── 02_svm_classifier.ipynb
    ├── 03_decision_tree.ipynb
    ├── 04_adaboost_classifier.ipynb
    └── 05_model_comparison.ipynb

