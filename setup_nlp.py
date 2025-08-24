import subprocess
import sys
import os
from pathlib import Path

def install_spacy_model():
    """Download spaCy English model"""
    try:
        print("Installing spaCy English model...")
        result = subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("[SUCCESS] spaCy English model installed successfully!")
            return True
        else:
            print(f"[ERROR] spaCy model installation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"[ERROR] Error installing spaCy model: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        print("Downloading NLTK data...")
        
        # Download required NLTK datasets
        datasets = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'vader_lexicon']
        
        for dataset in datasets:
            try:
                print(f"Downloading {dataset}...")
                nltk.download(dataset, quiet=True)
            except Exception as e:
                print(f"Warning: Failed to download {dataset}: {e}")
        
        print("[SUCCESS] NLTK data downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error downloading NLTK data: {e}")
        return False

def test_imports():
    """Test if all NLP libraries can be imported"""
    try:
        print("Testing imports...")
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("[SUCCESS] spaCy working correctly!")
        
        import nltk
        print("[SUCCESS] NLTK imported successfully!")
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("[SUCCESS] scikit-learn imported successfully!")
        
        import numpy as np
        print("[SUCCESS] NumPy imported successfully!")
        
        import pandas as pd
        print("[SUCCESS] Pandas imported successfully!")
        
        from textblob import TextBlob
        print("[SUCCESS] TextBlob imported successfully!")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Import test failed: {e}")
        return False

def create_nlp_config():
    """Create configuration file for NLP settings"""
    config_content = """# NLP Configuration

# spaCy model to use
SPACY_MODEL=en_core_web_sm

# Topic modeling parameters
MIN_TOPIC_COHERENCE=0.3
MAX_TOPICS=10
MIN_SEGMENT_LENGTH=30
MAX_SEGMENT_LENGTH=300

# Timestamp generation settings
MIN_TIMESTAMP_GAP=5.0
SIMILARITY_THRESHOLD=0.7
CONFIDENCE_THRESHOLD=0.6
"""
    
    try:
        with open(".env.nlp", "w") as f:
            f.write(config_content)
        print("[SUCCESS] NLP configuration created in .env.nlp")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create NLP config: {e}")
        return False

if __name__ == "__main__":
    print("Setting up NLP libraries for AI Timestamp Generator...")
    print("=" * 50)
    
    success = True
    
    # Install spaCy model
    if not install_spacy_model():
        success = False
    
    # Download NLTK data
    if not download_nltk_data():
        success = False
    
    # Test imports
    if not test_imports():
        success = False
    
    # Create config
    if not create_nlp_config():
        success = False
    
    print("=" * 50)
    if success:
        print("[SUCCESS] NLP setup completed successfully!")
        print("You can now use advanced content analysis features.")
    else:
        print("[ERROR] NLP setup completed with errors.")
        print("Some features may not work correctly.")