# Machine Learning Portfolio

A collection of machine learning, deep learning, computer vision, and generative AI projects developed through academic and hands-on project work.

This portfolio demonstrates practical experience across time-series forecasting, neural networks, computer vision, natural-language interfaces, SQL generation, data preprocessing, model evaluation, and interactive machine-learning applications.

---

## Projects

### 1. Smart Waste Management – Time-Series Forecasting

A deep-learning project focused on forecasting waste-bin fill levels from time-series sensor data for smart waste-management applications.

The project explores multiple neural-network architectures and compares predicted bin levels against actual sensor observations.

**Key Areas**
- Time-series preprocessing
- Sensor-data analysis
- Feature and sequence preparation
- LSTM neural networks
- CNN-based sequence modeling
- Hybrid CNN-LSTM architectures
- Model training and evaluation
- Forecasting and prediction analysis

**Technologies:**  
Python, TensorFlow/Keras, Pandas, NumPy, Scikit-learn, Matplotlib

📁 **[View Project](./smart-waste-management/)**

---

### 2. Farm Insect Classification Using YOLOv8

A computer-vision application for identifying agricultural insects from uploaded images using a YOLOv8 classification model.

The project includes model training and an interactive Streamlit application where users can upload an insect image and receive the predicted insect class with confidence scores.

**Key Areas**
- Computer vision
- Image classification
- YOLOv8
- Transfer learning
- Image preprocessing
- Model inference
- Confidence-based predictions
- Interactive ML application development

**Technologies:**  
Python, YOLOv8, Ultralytics, PyTorch, OpenCV, Streamlit

📁 **[View Project](./farm-insect-yolov8/)**

---

### 3. BridgesAI – NLP-to-SQL Bridge Infrastructure Explorer

BridgesAI is a natural-language analytics application that allows users to ask questions about bridge infrastructure data without manually writing SQL.

The application interprets a natural-language question, determines its analytical intent, retrieves relevant database schema information, generates a safe SQLite query using an LLM, executes the query, and presents the results through an interactive Streamlit interface.

**Core Pipeline**

```text
Natural-Language Question
        |
        v
Domain Relevance Validation
        |
        v
Intent Classification
        |
        v
Geographic Entity Detection
        |
        v
Schema-Aware RAG
        |
        v
LLM SQL Generation
        |
        v
SQL Safety Validation
        |
        v
SQLite Execution
        |
        v
Results & Visualization
```

**Key Areas**
- Natural Language Processing
- NLP-to-SQL
- Retrieval-Augmented Generation (RAG)
- OpenAI embeddings
- LLM integration
- Intent classification
- Schema-aware retrieval
- SQL generation and validation
- Geographic entity matching
- Interactive data visualization

**Technologies:**  
Python, Streamlit, OpenAI API, SQLite, SQLAlchemy, Pandas, NumPy, RapidFuzz

📁 **[View Project](./bridge-nlp-to-sql/)**

---

## Technical Skills Demonstrated

### Machine Learning & Deep Learning

- Time-series forecasting
- LSTM neural networks
- CNN and hybrid CNN-LSTM architectures
- Computer vision
- YOLOv8
- Transfer learning
- Model training and evaluation

### Generative AI & NLP

- Large Language Model API integration
- Retrieval-Augmented Generation (RAG)
- Embeddings and semantic similarity
- Natural-language-to-SQL generation
- Intent classification
- Prompt engineering

### Data & Analytics

- Python
- Pandas
- NumPy
- SQL
- SQLite
- SQLAlchemy
- Data preprocessing
- Data validation
- Exploratory analysis
- Data visualization

### Application Development

- Streamlit
- Interactive ML applications
- Model inference workflows
- Environment-based API key management
- Git and GitHub

---

## Project Overview

| Project | ML Area | Primary Application |
|---|---|---|
| [Smart Waste Management](./smart-waste-management/) | Deep Learning / Time Series | Waste-bin fill-level forecasting |
| [Farm Insect Classification](./farm-insect-yolov8/) | Computer Vision | Agricultural insect classification |
| [BridgesAI](./bridge-nlp-to-sql/) | Generative AI / NLP | Natural-language data analytics |

---

## Repository Structure

```text
ML_Portfolio/
│
├── README.md
│
├── smart-waste-management/
│   ├── README.md
│   ├── src/
│   ├── data/
│   └── results/
│
├── farm-insect-yolov8/
│   ├── README.md
│   └── ...
│
└── bridge-nlp-to-sql/
    ├── app.py
    ├── README.md
    ├── requirements.txt
    └── data/      
```

---

## Additional Projects

Additional machine-learning, data science, and AI projects will be added as they are prepared for public portfolio presentation.

---

## About This Portfolio

This repository serves as a consolidated portfolio of my machine-learning and AI work.

The projects include academic and hands-on implementations covering different stages of the machine-learning lifecycle, including data preparation, model development, training, evaluation, inference, natural-language interaction, and interactive application development.

The portfolio is intended to demonstrate both foundational machine-learning knowledge and practical experience building end-to-end data and AI applications.
