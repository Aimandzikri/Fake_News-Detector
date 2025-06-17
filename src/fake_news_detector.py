# --- Core Libraries ---
import os
import re
import traceback
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import warnings

# --- Data Handling & ML ---
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# --- NLP Libraries ---
import nltk
import spacy
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Web App & Utilities ---
import gdown
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from transformers import pipeline as transformers_pipeline

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# --- Constants and Configuration ---
# =============================================================================

MODEL_DIR = Path("models")
DATA_DIR = Path("data")
PAC_MODEL_PATH = MODEL_DIR / "model.pkl"
PAC_VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
FAKE_NEWS_DATA_PATH = DATA_DIR / "Fake.csv"
TRUE_NEWS_DATA_PATH = DATA_DIR / "True.csv"

# Performance-optimized constants
MAX_TEXT_LENGTH = 512  # Optimal for most models
BATCH_SIZE = 8  # Conservative batch size for stability
EVAL_SAMPLE_LIMIT = 300  # Reduced for faster evaluation
MAX_FEATURES = 5000  # TF-IDF feature limit

class ModelType(Enum):
    PASSIVE_AGGRESSIVE = "Classic AI (Passive Aggressive)"
    DISTILBERT = "Modern AI (RoBERTa)"
    SPACY_RULES = "Rule-Based (spaCy)"

# =============================================================================
# --- Optimized Preprocessing ---
# =============================================================================

@st.cache_resource(show_spinner="Initializing NLP tools...")
def get_nlp_tools():
    """Initialize NLP tools once and cache them."""
    # Download NLTK data
    for data_id in ['punkt', 'stopwords', 'wordnet']:
        try:
            path = f'tokenizers/{data_id}' if data_id == 'punkt' else f'corpora/{data_id}'
            nltk.data.find(path)
        except LookupError:
            nltk.download(data_id, quiet=True)
    
    return {
        'lemmatizer': WordNetLemmatizer(),
        'stop_words': set(stopwords.words('english')),
        'word_pattern': re.compile(r'[^a-z\s]'),
        'space_pattern': re.compile(r'\s+')
    }

def preprocess_text_fast(text: str, tools: Dict) -> str:
    """Faster preprocessing with minimal operations."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    
    # Quick preprocessing for performance
    text = text.lower()[:MAX_TEXT_LENGTH * 3]  # Truncate early
    text = tools['word_pattern'].sub(' ', text)
    text = tools['space_pattern'].sub(' ', text).strip()
    
    # Simplified tokenization and filtering
    words = [word for word in text.split() 
             if len(word) > 2 and word not in tools['stop_words']]
    
    return ' '.join(words[:MAX_TEXT_LENGTH // 5])  # Limit word count

@st.cache_data(show_spinner="Loading dataset...")
def load_data_optimized():
    """Optimized data loading with minimal preprocessing."""
    try:
        fake_df = pd.read_csv(FAKE_NEWS_DATA_PATH).assign(label='fake')
        true_df = pd.read_csv(TRUE_NEWS_DATA_PATH).assign(label='true')
        
        # Combine and clean
        df = pd.concat([fake_df, true_df], ignore_index=True)
        df['text'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).str.strip()
        df = df[['text', 'label']].dropna().query('text != ""')
        
        # Sample for faster processing if dataset is large
        if len(df) > 20000:
            df = df.sample(n=20000, random_state=42)
        
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

@st.cache_data
def get_processed_splits(_df: pd.DataFrame, _tools: Dict):
    """Get preprocessed train/test splits."""
    # Preprocess text
    _df['processed_text'] = _df['text'].apply(lambda x: preprocess_text_fast(x, _tools))
    
    # Split data
    X_proc, y = _df['processed_text'], _df['label']
    X_raw = _df['text']
    
    return train_test_split(X_proc, X_raw, y, test_size=0.2, random_state=42, stratify=y)

# =============================================================================
# --- Optimized Fake News Detector ---
# =============================================================================

class OptimizedFakeNewsDetector:
    """Performance-optimized fake news detector."""
    
    def __init__(self):
        self.tools = get_nlp_tools()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.pac_model: Optional[PassiveAggressiveClassifier] = None
        self.model_trained = False
        self.distilbert_pipeline = None
        self.nlp = None
        
        # Load models lazily
        self._load_pac_model()
    
    def _load_pac_model(self):
        """Load or initialize PAC model."""
        if PAC_MODEL_PATH.exists() and PAC_VECTORIZER_PATH.exists():
            try:
                self.pac_model = joblib.load(PAC_MODEL_PATH)
                self.vectorizer = joblib.load(PAC_VECTORIZER_PATH)
                self.model_trained = True
            except Exception:
                self._init_pac_model()
        else:
            self._init_pac_model()
    
    def _init_pac_model(self):
        """Initialize untrained PAC model."""
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_df=0.7, 
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_features=MAX_FEATURES,  # Use constant for consistency
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents='ascii'
        )
        self.pac_model = PassiveAggressiveClassifier(
            max_iter=1000, 
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.model_trained = False
    
    @st.cache_resource(show_spinner="Loading Fake News Detection Model...")
    def _load_distilbert(_self):
        """Load RoBERTa model specifically trained for fake news detection."""
        try:
            # Try multiple models in order of preference
            models_to_try = [
                "hamzab/roberta-fake-news-classification",
                "jy46604790/Fake-News-Bert-Detect",
                "distilbert-base-uncased-finetuned-sst-2-english"  # Fallback
            ]
            
            device = 0 if torch.cuda.is_available() else -1
            
            for model_name in models_to_try:
                try:
                    pipeline = transformers_pipeline(
                        "text-classification", 
                        model=model_name, 
                        device=device,
                        truncation=True,
                        max_length=MAX_TEXT_LENGTH,
                        return_all_scores=False
                    )
                    # Test the pipeline with a simple input
                    test_result = pipeline("This is a test.")
                    return pipeline
                except Exception as e:
                    st.warning(f"Failed to load {model_name}: {str(e)}")
                    continue
            
            return None
            
        except Exception as e:
            st.error(f"Failed to load any transformer model: {str(e)}")
            return None
    
    @st.cache_resource(show_spinner="Loading spaCy model...")
    def _load_spacy(_self):
        """Load spaCy model with error handling."""
        try:
            # Try to load the model
            nlp = spacy.load("en_core_web_sm")
            return nlp
        except OSError:
            st.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            return None
        except Exception as e:
            st.error(f"Failed to load spaCy model: {str(e)}")
            return None
    
    def get_distilbert(self):
        """Lazy load DistilBERT model."""
        if self.distilbert_pipeline is None:
            self.distilbert_pipeline = self._load_distilbert()
        return self.distilbert_pipeline
    
    def get_spacy(self):
        """Lazy load spaCy model."""
        if self.nlp is None:
            self.nlp = self._load_spacy()
        return self.nlp
    
    def train_pac(self, X_train: pd.Series, y_train: pd.Series) -> Tuple[bool, str]:
        """Train PAC model with optimization."""
        try:
            with st.spinner("Training Classic AI model..."):
                self._init_pac_model()
                X_train_tfidf = self.vectorizer.fit_transform(X_train)
                self.pac_model.fit(X_train_tfidf, y_train)
                self.model_trained = True
                
                # Save models
                MODEL_DIR.mkdir(exist_ok=True)
                joblib.dump(self.pac_model, PAC_MODEL_PATH)
                joblib.dump(self.vectorizer, PAC_VECTORIZER_PATH)
                
            return True, "Model trained successfully!"
        except Exception as e:
            return False, f"Training failed: {str(e)}"
    
    def predict(self, text: str, model_type: ModelType) -> Tuple[str, float, str]:
        """Route prediction to appropriate model with error handling."""
        try:
            # Use value-based comparison to avoid enum caching issues
            if hasattr(model_type, 'value'):
                model_value = model_type.value
                if model_value == "Classic AI (Passive Aggressive)":
                    return self._predict_pac(text)
                elif model_value == "Modern AI (RoBERTa)":
                    return self._predict_distilbert(text)
                elif model_value == "Rule-Based (spaCy)":
                    return self._predict_spacy(text)
                else:
                    return "Error", 0.0, f"Unknown model value: {model_value}"
            else:
                # Fallback for direct enum comparison
                if model_type == ModelType.PASSIVE_AGGRESSIVE:
                    return self._predict_pac(text)
                elif model_type == ModelType.DISTILBERT:
                    return self._predict_distilbert(text)
                elif model_type == ModelType.SPACY_RULES:
                    return self._predict_spacy(text)
                else:
                    return "Error", 0.0, f"Unknown model type: {model_type}"
        except Exception as e:
            return "Error", 0.0, f"Prediction failed: {str(e)}"
    
    def _predict_pac(self, text: str) -> Tuple[str, float, str]:
        """PAC model prediction."""
        if not self.model_trained:
            return "Error", 0.0, "Classic AI model not trained. Please train it first."
        
        try:
            processed_text = preprocess_text_fast(text, self.tools)
            text_tfidf = self.vectorizer.transform([processed_text])
            prediction = self.pac_model.predict(text_tfidf)[0]
            decision_score = self.pac_model.decision_function(text_tfidf)[0]
            
            confidence = 1 / (1 + np.exp(-abs(decision_score)))
            label = "Real" if prediction == 'true' else "Fake"
            explanation = f"**{label}** with {confidence:.1%} confidence using word frequency analysis."
            
            return label, confidence, explanation
        except Exception as e:
            return "Error", 0.0, f"PAC prediction failed: {str(e)}"
    
    def _predict_distilbert(self, text: str) -> Tuple[str, float, str]:
        """DistilBERT prediction with improved error handling."""
        pipeline = self.get_distilbert()
        if not pipeline:
            return "Error", 0.0, "Modern AI model not available. Please check model installation."
        
        try:
            # Truncate and clean text
            clean_text = text[:MAX_TEXT_LENGTH].strip()
            if not clean_text:
                return "Error", 0.0, "Empty text provided."
            
            result = pipeline(clean_text)
            
            # Handle different model output formats
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            label_text = result.get('label', '').upper()
            score = result.get('score', 0.5)
            
            # Map different label formats to our standard
            if any(fake_indicator in label_text for fake_indicator in ['FAKE', 'NEGATIVE', 'LABEL_0', '0']):
                label, confidence = "Fake", score
            elif any(real_indicator in label_text for real_indicator in ['REAL', 'TRUE', 'POSITIVE', 'LABEL_1', '1']):
                label, confidence = "Real", score
            else:
                # Default mapping based on score
                if score > 0.5:
                    label, confidence = "Real", score
                else:
                    label, confidence = "Fake", 1 - score
            
            explanation = f"**{label}** with {confidence:.1%} confidence using advanced language understanding."
            return label, confidence, explanation
            
        except Exception as e:
            return "Error", 0.0, f"Modern AI prediction failed: {str(e)}"
    
    def _predict_spacy(self, text: str) -> Tuple[str, float, str]:
        """Rule-based prediction with improved logic and error handling."""
        nlp = self.get_spacy()
        if not nlp:
            return "Error", 0.0, "spaCy model not available. Please install: python -m spacy download en_core_web_sm"

        try:
            # Ensure the input is always a string and not empty
            text = str(text).strip()
            if not text:
                return "Error", 0.0, "Empty text provided."

            # Enhanced keyword detection for fake news indicators
            fake_indicators = {
                'sensational': {'breaking', 'shocking', 'unbelievable', 'miracle', 'secret', 'exclusive'},
                'uncertain': {'allegedly', 'reportedly', 'sources say', 'rumors', 'claims', 'supposedly'},
                'emotional': {'outrage', 'scandal', 'exposed', 'truth', 'lies', 'hate', 'destroy'},
                'clickbait': {'you won\'t believe', 'shocking truth', 'they don\'t want', 'this will', 'must see'}
            }
            
            text_lower = text.lower()
            found_indicators = []
            category_scores = {}
            
            # Search for indicators in each category
            for category, keywords in fake_indicators.items():
                found = [kw for kw in keywords if kw in text_lower]
                if found:
                    found_indicators.extend(found)
                    category_scores[category] = len(found)
            
            # Calculate confidence based on number and type of indicators
            total_indicators = len(found_indicators)
            category_count = len(category_scores)
            
            # Determine prediction based on indicators found
            if total_indicators >= 3 or category_count >= 3:
                confidence = min(0.85, 0.6 + total_indicators * 0.05)
                explanation = f"**Fake** with {confidence:.1%} confidence. Found {total_indicators} indicators across {category_count} categories."
                return "Fake", confidence, explanation
            elif total_indicators >= 2:
                confidence = min(0.75, 0.55 + total_indicators * 0.05)
                explanation = f"**Fake** with {confidence:.1%} confidence. Found indicators: {', '.join(found_indicators[:3])}."
                return "Fake", confidence, explanation
            elif found_indicators:
                confidence = 0.65
                explanation = f"**Fake** with {confidence:.1%} confidence. Found indicator: {found_indicators[0]}."
                return "Fake", confidence, explanation
            else:
                confidence = 0.6
                explanation = f"**Real** with {confidence:.1%} confidence. No obvious fake indicators detected."
                return "Real", confidence, explanation

        except Exception as e:
            # Clean error handling without verbose logging
            return "Error", 0.0, f"Rule-based prediction failed: {str(e)[:100]}"

# =============================================================================
# --- Optimized UI Components ---
# =============================================================================

def display_prediction_result(label: str, confidence: float, explanation: str, model_name: str):
    """Display prediction result with improved formatting."""
    if label == "Error":
        st.error(f"❌ {model_name}: {explanation}")
        return
    
    is_fake = label.lower() == 'fake'
    color = "#e74c3c" if is_fake else "#2ecc71"
    icon = "🚨" if is_fake else "✅"
    
    st.markdown(f"""
    <div style="border: 2px solid {color}; padding: 1rem; border-radius: 10px; 
                background: linear-gradient(135deg, {color}15, {color}05); margin: 1rem 0;">
        <h4 style="color: {color}; margin: 0;">{icon} {model_name}</h4>
        <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>{explanation}</strong></p>
        <div style="background: #f0f0f0; border-radius: 10px; overflow: hidden;">
            <div style="width: {confidence*100:.1f}%; background: {color}; color: white; 
                        text-align: center; padding: 0.3rem; font-weight: bold;">
                {confidence:.1%}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def evaluate_model_fast(detector: OptimizedFakeNewsDetector, model_type: ModelType, 
                       X_test: pd.Series, y_test: pd.Series) -> Optional[List[str]]:
    """Fast model evaluation with improved error handling."""
    predictions = []
    
    try:
        if model_type == ModelType.PASSIVE_AGGRESSIVE:
            if not detector.model_trained:
                st.error("PAC model not trained")
                return None
            
            processed_texts = [preprocess_text_fast(text, detector.tools) for text in X_test]
            X_test_tfidf = detector.vectorizer.transform(processed_texts)
            predictions = detector.pac_model.predict(X_test_tfidf).tolist()
        
        elif model_type == ModelType.DISTILBERT:
            pipeline = detector.get_distilbert()
            if not pipeline:
                st.error("Modern AI model not available")
                return None
            
            progress = st.progress(0, text="Evaluating Modern AI...")
            texts = X_test.tolist()
            
            # Process individually to avoid batch issues
            for i, text in enumerate(texts):
                try:
                    clean_text = text[:MAX_TEXT_LENGTH].strip()
                    if not clean_text:
                        predictions.append('fake')  # Default for empty
                        continue
                        
                    result = pipeline(clean_text)
                    if isinstance(result, list) and len(result) > 0:
                        result = result[0]
                    
                    label_text = result.get('label', '').upper()
                    
                    # Map labels consistently
                    if any(fake_indicator in label_text for fake_indicator in ['FAKE', 'NEGATIVE', 'LABEL_0', '0']):
                        predictions.append('fake')
                    else:
                        predictions.append('true')
                        
                except Exception as e:
                    st.warning(f"Error processing text {i}: {str(e)}")
                    predictions.append('fake')  # Default fallback
                
                # Update progress
                if i % 10 == 0:  # Update every 10 items
                    progress.progress((i + 1) / len(texts))
            
            progress.empty()
        
        elif model_type == ModelType.SPACY_RULES:
            progress = st.progress(0, text="Evaluating Rule-Based model...")
            
            for i, text in enumerate(X_test):
                try:
                    label, _, _ = detector.predict(text, model_type)
                    if label == "Error":
                        predictions.append('fake')  # Default fallback
                    else:
                        predictions.append('fake' if label == 'Fake' else 'true')
                except Exception as e:
                    st.warning(f"Error in rule-based prediction {i}: {str(e)}")
                    predictions.append('fake')  # Default fallback
                
                # Update progress
                if i % 10 == 0:
                    progress.progress((i + 1) / len(X_test))
            
            progress.empty()
        
        return predictions
        
    except Exception as e:
        st.error(f"Evaluation failed for {model_type.value}: {str(e)}")
        return None

def display_metrics(model_name: str, y_true: pd.Series, y_pred: List[str]):
    """Display evaluation metrics efficiently."""
    try:
        accuracy = accuracy_score(y_true, y_pred)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(f"{model_name} Accuracy", f"{accuracy:.1%}")
        
        with col2:
            # Quick confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=['fake', 'true'])
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'], ax=ax)
            ax.set_title(f'{model_name} Confusion Matrix')
            st.pyplot(fig)
            plt.close()
            
    except Exception as e:
        st.error(f"Failed to display metrics for {model_name}: {str(e)}")

# =============================================================================
# --- Streamlit App ---
# =============================================================================

@st.cache_resource(show_spinner="Initializing application...")
def initialize_app():
    """Initialize the application."""
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    
    # Download data if needed
    load_dotenv()
    if not (FAKE_NEWS_DATA_PATH.exists() and TRUE_NEWS_DATA_PATH.exists()):
        FAKE_URL = os.getenv('FAKE_NEWS_URL', "https://drive.google.com/uc?export=download&id=1MhYVmHG3qW3sPGWcdYsWSbwNeOMK_7VQ")
        TRUE_URL = os.getenv('TRUE_NEWS_URL', "https://drive.google.com/uc?export=download&id=1aNLaFrUYI6KlaaI0-s42nWz3eX3lMuP3")
        
        with st.spinner("Downloading datasets..."):
            try:
                gdown.download(FAKE_URL, str(FAKE_NEWS_DATA_PATH), quiet=False)
                gdown.download(TRUE_URL, str(TRUE_NEWS_DATA_PATH), quiet=False)
            except Exception as e:
                st.error(f"Download failed: {e}")
                st.stop()
    
    return OptimizedFakeNewsDetector()

def main():
    """Main application."""
    st.set_page_config(
        page_title="AI Fake News Detector", 
        page_icon="🔍", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Add cache clearing button in sidebar for debugging
    with st.sidebar:
        if st.button("🔄 Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    
    detector = initialize_app()
    
    st.title("🔍 AI-Powered Fake News Detector")
    st.markdown("*Analyze news articles using multiple AI approaches for reliable detection.*")
    
    # Add model status indicators
    with st.expander("📊 Model Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pac_status = "✅ Ready" if detector.model_trained else "⚠️ Needs Training"
            st.write(f"**Classic AI:** {pac_status}")
        
        with col2:
            modern_status = "✅ Ready" if detector.get_distilbert() is not None else "❌ Not Available"
            st.write(f"**Modern AI:** {modern_status}")
        
        with col3:
            spacy_status = "✅ Ready" if detector.get_spacy() is not None else "❌ Not Available"
            st.write(f"**Rule-Based:** {spacy_status}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🚀 Quick Analysis", "📊 Compare Models", "⚡ Fast Evaluation"])
    
    with tab1:
        st.header("Quick Fake News Detection")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_input = st.text_area("Paste your news article here:", height=200, 
                                    placeholder="Enter the news text you want to analyze...")
        
        with col2:
            model_choice = st.selectbox("Choose AI Model:", [m.value for m in ModelType])
            analyze_btn = st.button("🔍 Analyze Article", type="primary", use_container_width=True)
        
        if analyze_btn and text_input.strip():
            try:
                model_type = next(m for m in ModelType if m.value == model_choice)
                with st.spinner(f"Analyzing with {model_choice}..."):
                    label, conf, expl = detector.predict(text_input, model_type)
                    st.markdown("---")
                    display_prediction_result(label, conf, expl, model_choice)
            except StopIteration:
                st.error(f"Invalid model choice: {model_choice}")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
        elif analyze_btn:
            st.warning("Please enter some text to analyze.")
    
    with tab2:
        st.header("Compare All AI Models")
        text_input = st.text_area("Enter news text for comparison:", height=150, key="compare_text")
        
        if st.button("🔍 Compare All Models", type="primary"):
            if text_input.strip():
                st.markdown("---")
                cols = st.columns(len(ModelType))
                
                for col, model_type in zip(cols, ModelType):
                    with col:
                        with st.spinner(f"Running {model_type.value}..."):
                            label, conf, expl = detector.predict(text_input, model_type)
                            display_prediction_result(label, conf, expl, model_type.value)
            else:
                st.warning("Please enter text to analyze.")
    
    with tab3:
        st.header("Fast Model Evaluation")
        st.info("Compare model performance on test data. Optimized for speed.")
        
        eval_size = st.slider("Test samples:", 50, EVAL_SAMPLE_LIMIT, 200, step=50)
        
        if st.button("⚡ Run Fast Evaluation", type="primary"):
            try:
                df = load_data_optimized()
                tools = detector.tools
                X_train_proc, X_test_proc, X_train_raw, X_test_raw, y_train, y_test = get_processed_splits(df, tools)
                
                # Sample for evaluation
                test_sample = min(eval_size, len(y_test))
                sample_idx = y_test.sample(n=test_sample, random_state=42).index
                X_test_sample = X_test_raw.loc[sample_idx]
                y_test_sample = y_test.loc[sample_idx]
                
                st.success(f"Testing on {test_sample} samples")
                st.markdown("---")
                
                # Evaluate models
                for model_type in ModelType:
                    with st.expander(f"📊 {model_type.value} Results", expanded=True):
                        if model_type == ModelType.PASSIVE_AGGRESSIVE:
                            if not detector.model_trained:
                                success, msg = detector.train_pac(X_train_proc, y_train)
                                if not success:
                                    st.error(f"Training failed: {msg}")
                                    continue
                        
                        predictions = evaluate_model_fast(detector, model_type, X_test_sample, y_test_sample)
                        if predictions and len(predictions) == len(y_test_sample):
                            display_metrics(model_type.value, y_test_sample, predictions)
                        else:
                            st.error(f"{model_type.value} evaluation failed or returned incomplete results")
                
                st.balloons()
                
            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")
                st.exception(e)  # Show full traceback for debugging

if __name__ == "__main__":
    main()
