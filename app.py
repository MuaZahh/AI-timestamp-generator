from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.utils import secure_filename
from decouple import config
import os
import uuid
import json
from datetime import datetime
import threading
from audio_processor import AudioProcessor
from speech_processor import SpeechProcessor
from content_analyzer import ContentAnalyzer
from timestamp_generator import TimestampGenerator
from export_processor import ExportProcessor
from batch_processor import batch_processor
from logging_config import initialize_logging, log_processing_start, log_processing_step, log_processing_error, log_processing_complete, PerformanceLogger
from error_handlers import register_error_handlers, validate_video_file, validate_video_exists, validate_batch_request, validate_export_format, handle_processing_errors, handle_file_operations, APIError, ValidationError, ProcessingError, FileError
import time

app = Flask(__name__)
CORS(app)

# Initialize logging
logging_config = initialize_logging("ai_timestamp_generator", 
                                  config('LOG_LEVEL', default='INFO'))
logger = logging_config.get_logger(__name__)
performance_logger = PerformanceLogger(logger)

# Register error handlers
register_error_handlers(app)

# Configuration
app.config['SECRET_KEY'] = config('SECRET_KEY', default='dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = config('DATABASE_URL', default='sqlite:///timestamp_generator.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = config('MAX_CONTENT_LENGTH', default=100 * 1024 * 1024, cast=int)  # 100MB
app.config['UPLOAD_FOLDER'] = config('UPLOAD_FOLDER', default='uploads')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

# Initialize processors
audio_processor = AudioProcessor()
speech_processor = SpeechProcessor()
content_analyzer = ContentAnalyzer()
timestamp_generator = TimestampGenerator()
export_processor = ExportProcessor()

# Database Models
class Video(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False, index=True)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    duration = db.Column(db.Float)
    status = db.Column(db.String(50), default='uploaded', index=True)  # uploaded, processing, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    transcript = db.relationship('Transcript', backref='video', uselist=False, cascade='all, delete-orphan')
    timestamps = db.relationship('Timestamp', backref='video', cascade='all, delete-orphan')

class Transcript(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    video_id = db.Column(db.String(36), db.ForeignKey('video.id'), nullable=False, index=True)
    text = db.Column(db.Text, nullable=False)
    confidence = db.Column(db.Float)
    language = db.Column(db.String(10))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

class Timestamp(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    video_id = db.Column(db.String(36), db.ForeignKey('video.id'), nullable=False, index=True)
    start_time = db.Column(db.Float, nullable=False, index=True)
    end_time = db.Column(db.Float)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

# Helper Functions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'm4v', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_size(file_path):
    return os.path.getsize(file_path)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/videos')
def videos():
    return render_template('videos.html')

@app.route('/video/<video_id>')
def video_detail(video_id):
    return render_template('video.html')

@app.route('/batch')
def batch_processing():
    return render_template('batch.html')

@app.route('/api')
def api_info():
    return jsonify({
        'message': 'AI Timestamp Generator API',
        'version': '1.0.0',
        'endpoints': {
            'upload': '/api/upload',
            'status': '/api/video/<video_id>/status',
            'process': '/api/video/<video_id>/process',
            'timestamps': '/api/video/<video_id>/timestamps'
        }
    })

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not supported. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Secure the filename
        original_filename = file.filename
        filename = secure_filename(file.filename)
        
        # Generate unique filename to prevent conflicts
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(file_path)
        
        # Get file size
        file_size = get_file_size(file_path)
        
        # Create database entry
        video = Video(
            filename=unique_filename,
            original_filename=original_filename,
            file_path=file_path,
            file_size=file_size,
            status='uploaded'
        )
        
        db.session.add(video)
        db.session.commit()
        
        return jsonify({
            'message': 'File uploaded successfully',
            'video_id': video.id,
            'filename': original_filename,
            'file_size': file_size,
            'status': video.status
        }), 201
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/video/<video_id>/status', methods=['GET'])
def get_video_status(video_id):
    video = Video.query.get_or_404(video_id)
    
    response_data = {
        'id': video.id,
        'filename': video.original_filename,
        'status': video.status,
        'file_size': video.file_size,
        'duration': video.duration,
        'created_at': video.created_at.isoformat(),
        'has_transcript': video.transcript is not None,
        'timestamp_count': len(video.timestamps)
    }
    
    return jsonify(response_data)

@app.route('/api/videos', methods=['GET'])
def list_videos():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    status_filter = request.args.get('status')
    
    # Limit per_page to prevent excessive queries
    per_page = min(per_page, 100)
    
    query = Video.query
    
    # Apply status filter if provided
    if status_filter:
        query = query.filter_by(status=status_filter)
    
    # Use pagination for better performance
    videos_paginated = query.order_by(Video.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    videos = videos_paginated.items
    
    video_list = []
    for video in videos:
        video_list.append({
            'id': video.id,
            'filename': video.original_filename,
            'status': video.status,
            'file_size': video.file_size,
            'duration': video.duration,
            'created_at': video.created_at.isoformat(),
            'has_transcript': video.transcript is not None,
            'timestamp_count': len(video.timestamps)
        })
    
    return jsonify({
        'videos': video_list,
        'pagination': {
            'page': videos_paginated.page,
            'pages': videos_paginated.pages,
            'per_page': videos_paginated.per_page,
            'total': videos_paginated.total,
            'has_next': videos_paginated.has_next,
            'has_prev': videos_paginated.has_prev
        }
    })

@app.route('/api/video/<video_id>', methods=['DELETE'])
def delete_video(video_id):
    video = Video.query.get_or_404(video_id)
    
    try:
        # Delete the physical file
        if os.path.exists(video.file_path):
            os.remove(video.file_path)
        
        # Delete from database (cascade will handle related records)
        db.session.delete(video)
        db.session.commit()
        
        return jsonify({'message': 'Video deleted successfully'})
        
    except Exception as e:
        return jsonify({'error': f'Delete failed: {str(e)}'}), 500

@app.route('/api/video/<video_id>/process', methods=['POST'])
def process_video(video_id):
    """Start video processing (audio extraction + transcription)"""
    video = Video.query.get_or_404(video_id)
    
    if video.status not in ['uploaded', 'failed']:
        return jsonify({'error': f'Video is already {video.status}'}), 400
    
    # Start processing in background thread
    def process_video_background():
        start_time = time.time()
        
        try:
            log_processing_start(logger, video.id, video.original_filename)
            
            # Update status to processing
            video.status = 'processing'
            db.session.commit()
            logger.info(f"Video {video.id}: Status updated to processing")
            
            # Step 1: Extract audio
            step_start = time.time()
            audio_path = audio_processor.extract_audio(video.file_path)
            step_duration = time.time() - step_start
            log_processing_step(logger, video.id, "audio extraction", step_duration)
            
            # Step 2: Get video info
            step_start = time.time()
            video_info = audio_processor.get_video_info(video.file_path)
            video.duration = video_info.get('duration')
            db.session.commit()
            step_duration = time.time() - step_start
            log_processing_step(logger, video.id, "video info extraction", step_duration)
            
            # Step 3: Transcribe audio
            step_start = time.time()
            transcription_result = speech_processor.transcribe_audio(audio_path)
            step_duration = time.time() - step_start
            log_processing_step(logger, video.id, "speech transcription", step_duration)
            
            # Step 4: Save transcript
            step_start = time.time()
            transcript = Transcript(
                video_id=video.id,
                text=transcription_result['transcript'],
                confidence=transcription_result['confidence'],
                language=transcription_result['language']
            )
            db.session.add(transcript)
            db.session.commit()
            step_duration = time.time() - step_start
            log_processing_step(logger, video.id, "transcript saving", step_duration)
            
            # Step 5: Perform content analysis
            step_start = time.time()
            content_analysis = content_analyzer.analyze_content(transcription_result)
            step_duration = time.time() - step_start
            log_processing_step(logger, video.id, "content analysis", step_duration)
            
            # Step 6: Generate intelligent timestamps
            step_start = time.time()
            intelligent_timestamps = timestamp_generator.generate_intelligent_timestamps(
                transcription_result, content_analysis
            )
            step_duration = time.time() - step_start
            log_processing_step(logger, video.id, f"timestamp generation ({len(intelligent_timestamps)} timestamps)", step_duration)
            
            # Step 7: Save intelligent timestamps
            step_start = time.time()
            for ts_data in intelligent_timestamps:
                timestamp = Timestamp(
                    video_id=video.id,
                    start_time=ts_data['start_time'],
                    end_time=ts_data['end_time'],
                    title=ts_data['title'],
                    description=ts_data['description'],
                    confidence=ts_data['confidence']
                )
                db.session.add(timestamp)
            db.session.commit()
            step_duration = time.time() - step_start
            log_processing_step(logger, video.id, "timestamp saving", step_duration)
            
            # Update status to completed
            video.status = 'completed'
            db.session.commit()
            
            # Clean up temporary audio file
            audio_processor.cleanup_temp_audio(audio_path)
            
            # Log completion
            total_duration = time.time() - start_time
            log_processing_complete(logger, video.id, total_duration)
            
        except Exception as e:
            log_processing_error(logger, video.id, "processing", e)
            video.status = 'failed'
            db.session.commit()
    
    # Start background thread
    thread = threading.Thread(target=process_video_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'message': 'Video processing started',
        'video_id': video.id,
        'status': 'processing'
    })

@app.route('/api/video/<video_id>/transcript', methods=['GET'])
def get_transcript(video_id):
    """Get transcript for a video"""
    video = Video.query.get_or_404(video_id)
    
    if not video.transcript:
        return jsonify({'error': 'Transcript not available'}), 404
    
    return jsonify({
        'video_id': video.id,
        'transcript': video.transcript.text,
        'confidence': video.transcript.confidence,
        'language': video.transcript.language,
        'created_at': video.transcript.created_at.isoformat()
    })

@app.route('/api/video/<video_id>/timestamps', methods=['GET'])
def get_timestamps(video_id):
    """Get timestamps for a video"""
    video = Video.query.get_or_404(video_id)
    
    # Use database-level sorting for better performance
    timestamps_query = Timestamp.query.filter_by(video_id=video_id).order_by(Timestamp.start_time)
    timestamps_data = timestamps_query.all()
    
    timestamps = []
    for ts in timestamps_data:
        timestamps.append({
            'id': ts.id,
            'start_time': ts.start_time,
            'end_time': ts.end_time,
            'title': ts.title,
            'description': ts.description,
            'confidence': ts.confidence
        })
    
    return jsonify({
        'video_id': video.id,
        'timestamps': timestamps
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db.session.execute('SELECT 1')
        
        # Test processors
        ffmpeg_ok = audio_processor.test_ffmpeg()
        deepgram_ok = speech_processor.test_api_connection()
        
        # Test NLP components
        nlp_ok = True
        try:
            content_analyzer.nlp("Test sentence")
        except:
            nlp_ok = False
        
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'ffmpeg': 'working' if ffmpeg_ok else 'error',
            'deepgram': 'connected' if deepgram_ok else 'error',
            'content_analysis': 'working' if nlp_ok else 'error'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/video/<video_id>/export/<format_type>', methods=['GET'])
def export_timestamps(video_id, format_type):
    """Export timestamps in specified format"""
    video = Video.query.get_or_404(video_id)
    
    if len(video.timestamps) == 0:
        return jsonify({'error': 'No timestamps available for export'}), 400
    
    # Get available formats
    available_formats = export_processor.get_available_formats()
    if format_type not in available_formats:
        return jsonify({
            'error': f'Unsupported format: {format_type}',
            'available_formats': list(available_formats.keys())
        }), 400
    
    try:
        # Convert timestamps to list of dictionaries
        timestamps_data = []
        for ts in video.timestamps:
            timestamps_data.append({
                'id': ts.id,
                'start_time': ts.start_time,
                'end_time': ts.end_time,
                'title': ts.title,
                'description': ts.description,
                'confidence': ts.confidence
            })
        
        # Prepare export data
        video_metadata = {
            'id': video.id,
            'filename': video.original_filename,
            'duration': video.duration,
            'file_size': video.file_size,
            'created_at': video.created_at.isoformat(),
            'status': video.status
        }
        
        # Export in requested format
        if format_type == 'youtube':
            content = export_processor.export_youtube_chapters(
                timestamps_data, 
                video_title=video.original_filename
            )
        elif format_type == 'json':
            content = export_processor.export_json(
                timestamps_data, 
                video_metadata=video_metadata
            )
        else:
            content = export_processor.export_format(format_type, timestamps_data)
        
        format_info = available_formats[format_type]
        
        response_data = {
            'format': format_type,
            'format_name': format_info['name'],
            'content': content,
            'filename': f"{video.original_filename.rsplit('.', 1)[0]}_timestamps.{format_info['extension']}",
            'mime_type': format_info['mime_type'],
            'timestamp_count': len(timestamps_data),
            'exported_at': datetime.utcnow().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@app.route('/api/video/<video_id>/export/<format_type>/download', methods=['GET'])
def download_export(video_id, format_type):
    """Download exported timestamps as file"""
    video = Video.query.get_or_404(video_id)
    
    if len(video.timestamps) == 0:
        return jsonify({'error': 'No timestamps available for export'}), 400
    
    # Get available formats
    available_formats = export_processor.get_available_formats()
    if format_type not in available_formats:
        return jsonify({
            'error': f'Unsupported format: {format_type}',
            'available_formats': list(available_formats.keys())
        }), 400
    
    try:
        # Convert timestamps to list of dictionaries
        timestamps_data = []
        for ts in video.timestamps:
            timestamps_data.append({
                'id': ts.id,
                'start_time': ts.start_time,
                'end_time': ts.end_time,
                'title': ts.title,
                'description': ts.description,
                'confidence': ts.confidence
            })
        
        # Prepare export data
        video_metadata = {
            'id': video.id,
            'filename': video.original_filename,
            'duration': video.duration,
            'file_size': video.file_size,
            'created_at': video.created_at.isoformat(),
            'status': video.status
        }
        
        # Export in requested format
        if format_type == 'youtube':
            content = export_processor.export_youtube_chapters(
                timestamps_data, 
                video_title=video.original_filename
            )
        elif format_type == 'json':
            content = export_processor.export_json(
                timestamps_data, 
                video_metadata=video_metadata
            )
        else:
            content = export_processor.export_format(format_type, timestamps_data)
        
        format_info = available_formats[format_type]
        filename = f"{video.original_filename.rsplit('.', 1)[0]}_timestamps.{format_info['extension']}"
        
        # Create response with file download
        response = app.response_class(
            response=content,
            status=200,
            mimetype=format_info['mime_type']
        )
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        
        return response
        
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/api/export/formats', methods=['GET'])
def get_export_formats():
    """Get list of available export formats"""
    return jsonify({
        'formats': export_processor.get_available_formats()
    })

# Batch processing endpoints
@app.route('/api/batch/create', methods=['POST'])
def create_batch_job():
    """Create a new batch processing job"""
    data = request.get_json()
    
    if not data or 'video_ids' not in data:
        return jsonify({'error': 'video_ids required'}), 400
    
    video_ids = data['video_ids']
    if not isinstance(video_ids, list) or len(video_ids) == 0:
        return jsonify({'error': 'video_ids must be a non-empty list'}), 400
    
    # Validate video IDs exist
    existing_videos = Video.query.filter(Video.id.in_(video_ids)).all()
    existing_ids = {video.id for video in existing_videos}
    missing_ids = set(video_ids) - existing_ids
    
    if missing_ids:
        return jsonify({
            'error': f'Videos not found: {list(missing_ids)}',
            'missing_video_ids': list(missing_ids)
        }), 400
    
    try:
        job_id = batch_processor.create_batch_job(video_ids)
        return jsonify({
            'message': 'Batch job created successfully',
            'job_id': job_id,
            'video_count': len(video_ids)
        }), 201
        
    except Exception as e:
        return jsonify({'error': f'Failed to create batch job: {str(e)}'}), 500

@app.route('/api/batch/pending', methods=['POST'])
def create_batch_from_pending():
    """Create batch job from pending videos"""
    data = request.get_json() or {}
    limit = data.get('limit', 10)
    
    if limit > 50:
        return jsonify({'error': 'Maximum limit is 50 videos per batch'}), 400
    
    from batch_processor import create_batch_from_pending_videos
    
    job_id = create_batch_from_pending_videos(limit)
    
    if not job_id:
        return jsonify({'message': 'No pending videos found'}), 404
    
    return jsonify({
        'message': 'Batch job created from pending videos',
        'job_id': job_id
    }), 201

@app.route('/api/batch/retry-failed', methods=['POST'])
def retry_failed_videos():
    """Create batch job to retry failed videos"""
    from batch_processor import process_all_failed_videos
    
    job_id = process_all_failed_videos()
    
    if not job_id:
        return jsonify({'message': 'No failed videos found'}), 404
    
    return jsonify({
        'message': 'Batch job created to retry failed videos',
        'job_id': job_id
    }), 201

@app.route('/api/batch/<job_id>/status', methods=['GET'])
def get_batch_job_status(job_id):
    """Get status of a batch job"""
    status = batch_processor.get_job_status(job_id)
    
    if not status:
        return jsonify({'error': 'Batch job not found'}), 404
    
    return jsonify(status)

@app.route('/api/batch/<job_id>/cancel', methods=['POST'])
def cancel_batch_job(job_id):
    """Cancel a batch job"""
    success = batch_processor.cancel_job(job_id)
    
    if not success:
        return jsonify({'error': 'Batch job not found or cannot be cancelled'}), 400
    
    return jsonify({'message': 'Batch job cancelled successfully'})

@app.route('/api/batch/jobs', methods=['GET'])
def list_batch_jobs():
    """List recent batch jobs"""
    limit = request.args.get('limit', 20, type=int)
    if limit > 100:
        limit = 100
    
    jobs = batch_processor.list_jobs(limit)
    return jsonify({
        'jobs': jobs,
        'total': len(jobs)
    })

@app.route('/api/batch/stats', methods=['GET'])
def get_batch_stats():
    """Get batch processing statistics"""
    stats = batch_processor.get_statistics()
    return jsonify(stats)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)

# For Vercel deployment
def handler(request):
    return app(request.environ, lambda *args: None)