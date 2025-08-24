from flask import jsonify, request
import logging
import traceback
from werkzeug.exceptions import HTTPException
from sqlalchemy.exc import SQLAlchemyError
import os

logger = logging.getLogger(__name__)

class APIError(Exception):
    """Custom API Error with status code and message"""
    def __init__(self, message, status_code=400, payload=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.payload = payload

class ValidationError(APIError):
    """Validation error for request data"""
    def __init__(self, message, field=None):
        super().__init__(message, 400)
        self.field = field

class ProcessingError(APIError):
    """Error during video processing"""
    def __init__(self, message, video_id=None):
        super().__init__(message, 500)
        self.video_id = video_id

class FileError(APIError):
    """File-related errors"""
    def __init__(self, message, filename=None):
        super().__init__(message, 400)
        self.filename = filename

def register_error_handlers(app):
    """Register all error handlers with the Flask app"""
    
    @app.errorhandler(APIError)
    def handle_api_error(error):
        """Handle custom API errors"""
        logger.error(f"API Error: {error.message}", extra={
            'status_code': error.status_code,
            'url': request.url,
            'method': request.method,
            'payload': error.payload
        })
        
        response = {
            'error': error.message,
            'status_code': error.status_code
        }
        
        if error.payload:
            response.update(error.payload)
        
        return jsonify(response), error.status_code
    
    @app.errorhandler(ValidationError)
    def handle_validation_error(error):
        """Handle validation errors"""
        logger.warning(f"Validation Error: {error.message}", extra={
            'field': error.field,
            'url': request.url,
            'method': request.method
        })
        
        response = {
            'error': error.message,
            'type': 'validation_error'
        }
        
        if error.field:
            response['field'] = error.field
        
        return jsonify(response), 400
    
    @app.errorhandler(ProcessingError)
    def handle_processing_error(error):
        """Handle processing errors"""
        logger.error(f"Processing Error: {error.message}", extra={
            'video_id': error.video_id,
            'url': request.url
        }, exc_info=True)
        
        response = {
            'error': error.message,
            'type': 'processing_error'
        }
        
        if error.video_id:
            response['video_id'] = error.video_id
        
        return jsonify(response), 500
    
    @app.errorhandler(FileError)
    def handle_file_error(error):
        """Handle file errors"""
        logger.error(f"File Error: {error.message}", extra={
            'filename': error.filename,
            'url': request.url
        })
        
        response = {
            'error': error.message,
            'type': 'file_error'
        }
        
        if error.filename:
            response['filename'] = error.filename
        
        return jsonify(response), 400
    
    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle 404 errors"""
        logger.warning(f"Not Found: {request.url}")
        
        return jsonify({
            'error': 'Resource not found',
            'url': request.url
        }), 404
    
    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        """Handle 405 errors"""
        logger.warning(f"Method Not Allowed: {request.method} {request.url}")
        
        return jsonify({
            'error': f'Method {request.method} not allowed for this endpoint',
            'allowed_methods': list(error.valid_methods) if hasattr(error, 'valid_methods') else []
        }), 405
    
    @app.errorhandler(413)
    def handle_file_too_large(error):
        """Handle file upload size limit errors"""
        logger.warning(f"File too large: {request.url}")
        
        max_size = app.config.get('MAX_CONTENT_LENGTH', 0)
        max_size_mb = max_size / 1024 / 1024 if max_size else 'unknown'
        
        return jsonify({
            'error': 'File too large',
            'max_size_mb': max_size_mb
        }), 413
    
    @app.errorhandler(SQLAlchemyError)
    def handle_database_error(error):
        """Handle database errors"""
        logger.error(f"Database Error: {str(error)}", exc_info=True)
        
        # Don't expose internal database errors in production
        if app.config.get('ENV') == 'production':
            message = 'A database error occurred'
        else:
            message = str(error)
        
        return jsonify({
            'error': message,
            'type': 'database_error'
        }), 500
    
    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        """Handle general HTTP exceptions"""
        logger.error(f"HTTP Exception: {error.code} {error.name}", extra={
            'url': request.url,
            'method': request.method
        })
        
        return jsonify({
            'error': error.description,
            'status_code': error.code
        }), error.code
    
    @app.errorhandler(500)
    def handle_internal_server_error(error):
        """Handle 500 internal server errors"""
        logger.error(f"Internal Server Error: {request.url}", exc_info=True)
        
        # Don't expose internal errors in production
        if app.config.get('ENV') == 'production':
            message = 'An internal server error occurred'
        else:
            message = str(error)
        
        return jsonify({
            'error': message,
            'type': 'internal_server_error'
        }), 500
    
    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        """Handle unexpected errors"""
        logger.critical(f"Unexpected Error: {str(error)}", exc_info=True)
        
        # Don't expose internal errors in production
        if app.config.get('ENV') == 'production':
            message = 'An unexpected error occurred'
        else:
            message = f'Unexpected error: {str(error)}'
        
        return jsonify({
            'error': message,
            'type': 'unexpected_error'
        }), 500

# Validation helpers
def validate_video_file(file):
    """Validate uploaded video file"""
    if not file:
        raise ValidationError("No file provided")
    
    if file.filename == '':
        raise ValidationError("No file selected")
    
    # Check file extension
    allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'm4v', 'webm'}
    if not ('.' in file.filename and 
            file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        raise ValidationError(
            f"File type not supported. Allowed types: {', '.join(allowed_extensions)}",
            field="file"
        )
    
    return True

def validate_video_exists(video):
    """Validate video exists in database"""
    if not video:
        raise APIError("Video not found", status_code=404)
    return True

def validate_batch_request(data):
    """Validate batch processing request"""
    if not data or 'video_ids' not in data:
        raise ValidationError("video_ids required", field="video_ids")
    
    video_ids = data['video_ids']
    if not isinstance(video_ids, list) or len(video_ids) == 0:
        raise ValidationError("video_ids must be a non-empty list", field="video_ids")
    
    if len(video_ids) > 50:
        raise ValidationError("Maximum 50 videos per batch", field="video_ids")
    
    return True

def validate_export_format(format_type, available_formats):
    """Validate export format"""
    if format_type not in available_formats:
        raise ValidationError(
            f"Unsupported format: {format_type}. Available formats: {', '.join(available_formats.keys())}",
            field="format"
        )
    return True

# Context managers for error handling
from contextlib import contextmanager

@contextmanager
def handle_processing_errors(video_id=None, operation="processing"):
    """Context manager for handling processing errors"""
    try:
        yield
    except FileNotFoundError as e:
        raise ProcessingError(f"Required file not found during {operation}: {str(e)}", video_id)
    except PermissionError as e:
        raise ProcessingError(f"Permission denied during {operation}: {str(e)}", video_id)
    except OSError as e:
        raise ProcessingError(f"System error during {operation}: {str(e)}", video_id)
    except Exception as e:
        logger.error(f"Unexpected error during {operation} for video {video_id}: {str(e)}", exc_info=True)
        raise ProcessingError(f"Unexpected error during {operation}", video_id)

@contextmanager
def handle_file_operations(filename=None):
    """Context manager for handling file operations"""
    try:
        yield
    except FileNotFoundError as e:
        raise FileError(f"File not found: {str(e)}", filename)
    except PermissionError as e:
        raise FileError(f"Permission denied: {str(e)}", filename)
    except OSError as e:
        raise FileError(f"File system error: {str(e)}", filename)
    except Exception as e:
        logger.error(f"Unexpected file error for {filename}: {str(e)}", exc_info=True)
        raise FileError("Unexpected file error occurred", filename)

# Monitoring and alerting
class ErrorMonitor:
    """Monitor and track application errors"""
    
    def __init__(self):
        self.error_counts = {}
        self.recent_errors = []
        self.max_recent_errors = 100
    
    def record_error(self, error_type, message, context=None):
        """Record an error occurrence"""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        error_record = {
            'type': error_type,
            'message': message,
            'context': context or {},
            'timestamp': logger.handlers[0].formatter.formatTime(logger.makeRecord(
                'error_monitor', logging.ERROR, __file__, 0, '', (), None
            ))
        }
        
        self.recent_errors.append(error_record)
        
        # Keep only recent errors
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors.pop(0)
    
    def get_error_stats(self):
        """Get error statistics"""
        return {
            'error_counts': self.error_counts,
            'recent_errors': self.recent_errors[-10:],  # Last 10 errors
            'total_errors': sum(self.error_counts.values())
        }
    
    def should_alert(self, error_type, threshold=10):
        """Check if error count exceeds threshold"""
        return self.error_counts.get(error_type, 0) >= threshold

# Global error monitor instance
error_monitor = ErrorMonitor()