# AI Timestamp Generator

A free AI-powered tool that automatically generates timestamps for any video by analyzing audio content and creating meaningful segments.

## Phase 5 Complete ✅

### Complete Export System Implemented:
- **Multiple Export Formats**:
  - YouTube Chapters (copy-paste ready for video descriptions)
  - SRT Subtitles (for video players and streaming)
  - WebVTT Subtitles (for HTML5 video players)
  - JSON Data (machine-readable for developers)
  - CSV Spreadsheet (for data analysis and Excel)
  - Plain Text (simple sharing format)

- **Export Features**:
  - One-click copy-to-clipboard for all formats
  - Direct file downloads with proper MIME types
  - Real-time export preview with formatting
  - Metadata inclusion (file size, timestamp count, export time)
  - Format-specific optimization and validation

### Full Web Application:
- **Modern Web Interface**: Responsive design with Bootstrap 5, drag-and-drop upload, real-time status
- **User Experience**: Visual processing steps, file validation, auto-refresh, bulk management

### AI Processing Pipeline:
- **Advanced Content Analysis**: 
  - Topic extraction using TF-IDF and K-means clustering
  - Semantic segmentation with sentence similarity
  - Named entity recognition and key phrase extraction
  - Sentiment analysis across content
  - Speaker pattern analysis
- **Intelligent Timestamp Generation**:
  - Multiple generation strategies (semantic boundaries, speaker changes, topic transitions)
  - Content engagement point detection
  - Timestamp optimization and merging
  - Confidence-based filtering

### Core Features:
- **Video Upload System**: Secure file upload with validation (MP4, AVI, MOV, MKV, WMV, FLV, M4V, WEBM)
- **Audio Extraction**: FFmpeg integration for extracting audio from any video format
- **Speech-to-Text**: Deepgram API integration with advanced features
- **Database Schema**: Complete data models for videos, transcripts, and timestamps
- **Background Processing**: Asynchronous video processing pipeline
- **RESTful API**: Complete API endpoints for all operations

### Tech Stack:
- **Backend**: Python + Flask
- **Database**: SQLAlchemy (SQLite for development)
- **Audio Processing**: FFmpeg
- **Speech-to-Text**: Deepgram API
- **NLP & AI**: spaCy, scikit-learn, NLTK, TextBlob
- **Content Analysis**: TF-IDF vectorization, K-means clustering
- **Storage**: Local filesystem

## Quick Start

### 1. Setup Environment
```bash
# Install Python dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
```

### 2. Configure Deepgram API
Edit `.env` file:
```
DEEPGRAM_API_KEY=your_deepgram_api_key_here
```
Get free $200 credit at: https://deepgram.com/

### 3. Install FFmpeg and NLP Libraries
```bash
# Run the automated installers
python install_ffmpeg.py
python setup_nlp.py
```

### 4. Start the Application
```bash
python app.py
```
Server runs on: http://localhost:5000

**🎉 The web application is now running!**
- Open http://localhost:5000 in your browser
- Upload a video file using drag-and-drop
- Watch the AI process your video automatically
- View generated intelligent timestamps

## API Endpoints

### Core Operations
- `GET /` - API information
- `GET /health` - System health check
- `POST /api/upload` - Upload video file
- `POST /api/video/{id}/process` - Start processing
- `GET /api/video/{id}/status` - Check processing status

### Results
- `GET /api/video/{id}/transcript` - Get full transcript
- `GET /api/video/{id}/timestamps` - Get generated timestamps
- `GET /api/videos` - List all videos
- `DELETE /api/video/{id}` - Delete video

### Example Usage
```bash
# 1. Upload video
curl -X POST -F "file=@video.mp4" http://localhost:5000/api/upload

# 2. Start processing (using video_id from upload response)
curl -X POST http://localhost:5000/api/video/{video_id}/process

# 3. Check status
curl http://localhost:5000/api/video/{video_id}/status

# 4. Get timestamps when completed
curl http://localhost:5000/api/video/{video_id}/timestamps

# 5. Export timestamps in different formats
curl http://localhost:5000/api/video/{video_id}/export/youtube
curl http://localhost:5000/api/video/{video_id}/export/srt
curl http://localhost:5000/api/video/{video_id}/export/json

# 6. Download export files
curl http://localhost:5000/api/video/{video_id}/export/youtube/download -o timestamps.txt
```

## Current Capabilities

### Audio Processing
- Extracts audio from any video format
- Converts to optimal format for transcription (16kHz WAV)
- Handles large files efficiently
- Temporary file management

### Speech Recognition
- High accuracy transcription (Deepgram Nova-2 model)
- Speaker identification and separation
- Multi-language support
- Confidence scoring

### Intelligent Timestamps
- **Semantic Boundaries**: Content-aware segment detection
- **Topic Transitions**: Automatic topic change identification  
- **Speaker Changes**: Multi-speaker content segmentation
- **Engagement Points**: Call-to-action and Q&A detection
- **Content Structure**: Logical section organization
- **Smart Titles**: AI-generated descriptive labels

### Data Storage
- Persistent video metadata
- Full transcript storage
- Generated timestamps with confidence scores
- Processing status tracking

## What's Next: Phase 6 (Final)
- Performance optimization and caching
- Batch processing for multiple videos
- Production deployment configuration
- Advanced timestamp editing features

## Project Structure
```
AI timestamp generator/
├── app.py                 # Main Flask application
├── audio_processor.py     # FFmpeg audio extraction
├── speech_processor.py    # Deepgram transcription
├── content_analyzer.py    # Advanced NLP content analysis
├── timestamp_generator.py # Intelligent timestamp generation
├── export_processor.py    # Multi-format export system
├── install_ffmpeg.py      # FFmpeg setup script
├── setup_nlp.py          # NLP libraries setup
├── requirements.txt       # Python dependencies
├── research_findings.md   # Phase 1 research
├── .env.example          # Environment template
├── .env.nlp              # NLP configuration
├── templates/            # HTML templates
├── static/               # CSS, JS, and assets
├── ffmpeg/               # Local FFmpeg installation
└── uploads/              # Video file storage
```

## System Requirements
- Python 3.7+
- Windows/Linux/Mac
- 4GB RAM minimum
- Internet connection for Deepgram API