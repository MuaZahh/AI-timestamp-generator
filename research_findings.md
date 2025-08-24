# Phase 1: Research & Technical Foundation - Findings

## Video Processing with FFmpeg

### Supported Input Formats
- **Video Containers**: MP4, MOV, AVI, MKV, WMV, FLV, and many others
- **Audio Extraction Methods**:
  - **Copy Mode**: No re-encoding (fastest, preserves quality)
  - **Re-encoding Mode**: Convert to specific formats (MP3, WAV, AAC)
- **Output Audio Formats**:
  - Lossy: MP3 (libmp3lame), AAC, OGG Vorbis, AC-3
  - Lossless: WAV (PCM), FLAC, PCM_s16le
- **Key Features**: Multiple audio stream support, 30-second chunk processing

## Speech-to-Text API Options (Free/Affordable)

### 1. OpenAI Whisper API
- **Accuracy**: 7.60% WER (best among major providers)
- **Free Tier**: 30 hours of credits for testing
- **Limitations**: 
  - Rate limiting (minutes-based)
  - No speaker diarization
  - 30-second audio chunks
  - Language-dependent performance
- **Cost**: More expensive than alternatives after free credits

### 2. Deepgram (Top Recommendation)
- **Performance**: #1 ranked speech-to-text API (2025)
- **Free Tier**: $200 credit
- **Speed**: ~20 seconds/hour batch transcription
- **Features**: Speaker diarization, real-time processing

### 3. Microsoft Azure Speech-to-Text
- **Free Tier**: 5 audio hours per month
- **Languages**: 85 supported
- **Features**: Batch transcription, real-time, language ID, diarization
- **Performance**: Good accuracy and speed combination

### 4. AssemblyAI
- **Ranking**: #5 in top APIs
- **Features**: Speaker diarization, sentiment analysis
- **Languages**: Speaker ID for 10 languages
- **Cost**: More expensive for high volume

### 5. Speechmatics
- **Languages**: 50+ supported
- **Free Tier**: 8 hours per month
- **Pricing**: $0.30/hour after free tier

## Content Analysis & Topic Detection Libraries

### 1. spaCy (Recommended for Production)
- **Performance**: Industrial-strength, optimized for speed
- **Features**: NER, POS tagging, dependency parsing, word vectors
- **Languages**: 49+ supported for tokenization
- **Integration**: Works with TensorFlow, PyTorch
- **Use Case**: Real-time processing, large-scale applications

### 2. NLTK (Good for Research/Education)
- **Resources**: 50+ corpora and lexical resources
- **Features**: Comprehensive preprocessing, classification, stemming
- **Best For**: Educational projects, research, prototyping

### 3. Gensim (Topic Modeling Specialist)
- **Specialty**: Topic modeling, document similarity
- **Algorithms**: LDA, word2vec, doc2vec
- **Use Cases**: Topic discovery, document clustering, similarity analysis
- **Memory**: Efficient for large document collections

### 4. Additional Tools
- **Textacy**: Extends spaCy for topic modeling
- **Embedded-topic-model**: Topics as points in embedding space
- **pyLDAvis**: Visualization for topic models

## Recommended Tech Stack

### Core Technologies
- **Backend**: Python (Flask/FastAPI) or Node.js
- **Video Processing**: FFmpeg
- **Speech-to-Text**: Deepgram (best balance) or Whisper (if budget allows)
- **NLP**: spaCy + Gensim combination
- **Database**: PostgreSQL
- **Storage**: Local filesystem or cloud storage (within free tiers)

### Architecture Approach
1. **Upload**: Secure file upload with validation
2. **Extract**: FFmpeg audio extraction (WAV format recommended)
3. **Transcribe**: Deepgram API for speech-to-text
4. **Analyze**: spaCy preprocessing + Gensim topic modeling
5. **Generate**: Smart timestamp creation based on content analysis
6. **Export**: Multiple formats (YouTube chapters, SRT, JSON)

### Cost Optimization for Free Service
- Use Deepgram's $200 credit efficiently
- Implement audio compression before transcription
- Cache transcription results
- Batch process multiple requests
- Use open-source models where possible (spaCy, Gensim)

## Next Steps for Phase 2
- Set up Python development environment
- Install and test FFmpeg integration
- Create Deepgram API account and test integration
- Design database schema for videos, transcripts, timestamps
- Build basic file upload system with validation