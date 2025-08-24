import os
import json
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import queue
import uuid

from audio_processor import AudioProcessor
from speech_processor import SpeechProcessor
from content_analyzer import ContentAnalyzer
from timestamp_generator import TimestampGenerator
from cache_manager import cache_manager

logger = logging.getLogger(__name__)

class BatchStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BatchJob:
    id: str
    video_ids: List[str]
    status: BatchStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_videos: int = 0
    processed_videos: int = 0
    failed_videos: int = 0
    current_video_id: Optional[str] = None
    error_message: Optional[str] = None
    progress_callback: Optional[Callable] = None

class BatchProcessor:
    def __init__(self, max_concurrent_jobs=2, max_workers_per_job=1):
        """
        Initialize batch processor
        
        Args:
            max_concurrent_jobs (int): Maximum number of batch jobs to run simultaneously
            max_workers_per_job (int): Maximum workers per batch job (usually 1 for video processing)
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_workers_per_job = max_workers_per_job
        
        self.job_queue = queue.Queue()
        self.active_jobs = {}
        self.completed_jobs = {}
        self.job_history = []
        
        # Processing components
        self.audio_processor = AudioProcessor()
        self.speech_processor = SpeechProcessor()
        self.content_analyzer = ContentAnalyzer()
        self.timestamp_generator = TimestampGenerator()
        
        # Worker thread management
        self.worker_threads = []
        self.shutdown_event = threading.Event()
        
        # Start worker threads
        self._start_workers()
        
        logger.info(f"Batch processor initialized with {max_concurrent_jobs} concurrent jobs")
    
    def create_batch_job(self, video_ids: List[str], progress_callback: Optional[Callable] = None) -> str:
        """
        Create a new batch processing job
        
        Args:
            video_ids (List[str]): List of video IDs to process
            progress_callback (Callable): Optional callback for progress updates
            
        Returns:
            str: Batch job ID
        """
        if not video_ids:
            raise ValueError("No video IDs provided")
        
        job_id = str(uuid.uuid4())
        
        batch_job = BatchJob(
            id=job_id,
            video_ids=video_ids,
            status=BatchStatus.PENDING,
            created_at=datetime.utcnow(),
            total_videos=len(video_ids),
            progress_callback=progress_callback
        )
        
        # Add to queue
        self.job_queue.put(batch_job)
        
        logger.info(f"Created batch job {job_id} with {len(video_ids)} videos")
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get status of a batch job"""
        # Check active jobs
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return self._job_to_dict(job)
        
        # Check completed jobs
        if job_id in self.completed_jobs:
            job = self.completed_jobs[job_id]
            return self._job_to_dict(job)
        
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a batch job if possible"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = BatchStatus.CANCELLED
            logger.info(f"Cancelled batch job {job_id}")
            return True
        
        return False
    
    def list_jobs(self, limit: int = 50) -> List[Dict]:
        """List recent batch jobs"""
        all_jobs = []
        
        # Add active jobs
        for job in self.active_jobs.values():
            all_jobs.append(self._job_to_dict(job))
        
        # Add completed jobs
        for job in self.completed_jobs.values():
            all_jobs.append(self._job_to_dict(job))
        
        # Sort by creation time (newest first)
        all_jobs.sort(key=lambda x: x['created_at'], reverse=True)
        
        return all_jobs[:limit]
    
    def _start_workers(self):
        """Start worker threads for processing batch jobs"""
        for i in range(self.max_concurrent_jobs):
            worker = threading.Thread(target=self._worker_loop, name=f"BatchWorker-{i}")
            worker.daemon = True
            worker.start()
            self.worker_threads.append(worker)
        
        logger.info(f"Started {len(self.worker_threads)} batch worker threads")
    
    def _worker_loop(self):
        """Main worker loop for processing batch jobs"""
        while not self.shutdown_event.is_set():
            try:
                # Get next job from queue (timeout prevents blocking forever)
                try:
                    job = self.job_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Move job to active jobs
                self.active_jobs[job.id] = job
                job.status = BatchStatus.PROCESSING
                job.started_at = datetime.utcnow()
                
                logger.info(f"Starting batch job {job.id} with {job.total_videos} videos")
                
                # Process the batch job
                self._process_batch_job(job)
                
                # Move job to completed jobs
                job.completed_at = datetime.utcnow()
                self.completed_jobs[job.id] = job
                del self.active_jobs[job.id]
                
                # Add to history for cleanup
                self.job_history.append(job.id)
                
                # Cleanup old completed jobs (keep last 100)
                if len(self.completed_jobs) > 100:
                    oldest_job_id = self.job_history.pop(0)
                    if oldest_job_id in self.completed_jobs:
                        del self.completed_jobs[oldest_job_id]
                
                self.job_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in batch worker: {e}")
                if 'job' in locals():
                    job.status = BatchStatus.FAILED
                    job.error_message = str(e)
                    job.completed_at = datetime.utcnow()
                    self.completed_jobs[job.id] = job
                    if job.id in self.active_jobs:
                        del self.active_jobs[job.id]
                
                time.sleep(1)  # Brief pause before continuing
    
    def _process_batch_job(self, job: BatchJob):
        """Process a single batch job"""
        from app import db, Video, Transcript, Timestamp  # Import here to avoid circular imports
        
        try:
            for i, video_id in enumerate(job.video_ids):
                # Check if job was cancelled
                if job.status == BatchStatus.CANCELLED:
                    logger.info(f"Batch job {job.id} was cancelled")
                    break
                
                job.current_video_id = video_id
                
                try:
                    # Get video from database
                    video = Video.query.get(video_id)
                    if not video:
                        logger.warning(f"Video {video_id} not found, skipping")
                        job.failed_videos += 1
                        continue
                    
                    # Skip if already processed
                    if video.status == 'completed':
                        logger.info(f"Video {video_id} already processed, skipping")
                        job.processed_videos += 1
                        self._update_progress(job)
                        continue
                    
                    # Check cache first
                    cached_transcript = cache_manager.get('transcripts', video_id)
                    cached_analysis = cache_manager.get('analysis', video_id)
                    
                    if cached_transcript and cached_analysis:
                        logger.info(f"Using cached data for video {video_id}")
                        
                        # Save cached transcript to database
                        if not video.transcript:
                            transcript = Transcript(
                                video_id=video.id,
                                text=cached_transcript['transcript'],
                                confidence=cached_transcript.get('confidence', 0.8),
                                language=cached_transcript.get('language', 'en')
                            )
                            db.session.add(transcript)
                        
                        # Generate timestamps from cached analysis
                        if len(video.timestamps) == 0:
                            intelligent_timestamps = self.timestamp_generator.generate_intelligent_timestamps(
                                cached_transcript, cached_analysis
                            )
                            
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
                        
                        video.status = 'completed'
                        db.session.commit()
                        
                    else:
                        # Full processing pipeline
                        logger.info(f"Processing video {video_id} from scratch")
                        
                        video.status = 'processing'
                        db.session.commit()
                        
                        # Step 1: Extract audio
                        audio_path = self.audio_processor.extract_audio(video.file_path)
                        
                        # Step 2: Get video info
                        video_info = self.audio_processor.get_video_info(video.file_path)
                        video.duration = video_info.get('duration')
                        db.session.commit()
                        
                        # Step 3: Transcribe audio
                        transcription_result = self.speech_processor.transcribe_audio(audio_path)
                        
                        # Step 4: Save transcript
                        if not video.transcript:
                            transcript = Transcript(
                                video_id=video.id,
                                text=transcription_result['transcript'],
                                confidence=transcription_result['confidence'],
                                language=transcription_result['language']
                            )
                            db.session.add(transcript)
                        
                        # Step 5: Content analysis
                        content_analysis = self.content_analyzer.analyze_content(transcription_result)
                        
                        # Step 6: Generate timestamps
                        if len(video.timestamps) == 0:
                            intelligent_timestamps = self.timestamp_generator.generate_intelligent_timestamps(
                                transcription_result, content_analysis
                            )
                            
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
                        
                        # Step 7: Cache results
                        cache_manager.set('transcripts', video_id, transcription_result)
                        cache_manager.set('analysis', video_id, content_analysis)
                        
                        video.status = 'completed'
                        db.session.commit()
                        
                        # Cleanup temporary audio file
                        self.audio_processor.cleanup_temp_audio(audio_path)
                    
                    job.processed_videos += 1
                    logger.info(f"Successfully processed video {video_id} ({i+1}/{job.total_videos})")
                    
                except Exception as e:
                    logger.error(f"Error processing video {video_id}: {e}")
                    job.failed_videos += 1
                    
                    # Update video status
                    try:
                        video = Video.query.get(video_id)
                        if video:
                            video.status = 'failed'
                            db.session.commit()
                    except:
                        pass
                
                # Update progress
                self._update_progress(job)
            
            # Mark job as completed
            if job.status != BatchStatus.CANCELLED:
                job.status = BatchStatus.COMPLETED
                logger.info(f"Batch job {job.id} completed: {job.processed_videos} processed, {job.failed_videos} failed")
        
        except Exception as e:
            logger.error(f"Fatal error in batch job {job.id}: {e}")
            job.status = BatchStatus.FAILED
            job.error_message = str(e)
    
    def _update_progress(self, job: BatchJob):
        """Update job progress and call callback if provided"""
        if job.progress_callback:
            try:
                progress_data = {
                    'job_id': job.id,
                    'total_videos': job.total_videos,
                    'processed_videos': job.processed_videos,
                    'failed_videos': job.failed_videos,
                    'current_video_id': job.current_video_id,
                    'progress_percent': round((job.processed_videos + job.failed_videos) / job.total_videos * 100, 2)
                }
                job.progress_callback(progress_data)
            except Exception as e:
                logger.warning(f"Error in progress callback: {e}")
    
    def _job_to_dict(self, job: BatchJob) -> Dict:
        """Convert BatchJob to dictionary for API responses"""
        return {
            'id': job.id,
            'status': job.status.value,
            'created_at': job.created_at.isoformat(),
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'total_videos': job.total_videos,
            'processed_videos': job.processed_videos,
            'failed_videos': job.failed_videos,
            'current_video_id': job.current_video_id,
            'progress_percent': round((job.processed_videos + job.failed_videos) / job.total_videos * 100, 2) if job.total_videos > 0 else 0,
            'error_message': job.error_message,
            'video_ids': job.video_ids
        }
    
    def shutdown(self):
        """Gracefully shutdown the batch processor"""
        logger.info("Shutting down batch processor...")
        
        self.shutdown_event.set()
        
        # Wait for workers to finish (with timeout)
        for worker in self.worker_threads:
            worker.join(timeout=5.0)
        
        logger.info("Batch processor shutdown complete")
    
    def get_statistics(self) -> Dict:
        """Get batch processing statistics"""
        total_jobs = len(self.active_jobs) + len(self.completed_jobs)
        
        completed_count = sum(1 for job in self.completed_jobs.values() 
                            if job.status == BatchStatus.COMPLETED)
        failed_count = sum(1 for job in self.completed_jobs.values() 
                         if job.status == BatchStatus.FAILED)
        cancelled_count = sum(1 for job in self.completed_jobs.values() 
                            if job.status == BatchStatus.CANCELLED)
        
        total_videos_processed = sum(job.processed_videos for job in self.completed_jobs.values())
        total_videos_failed = sum(job.failed_videos for job in self.completed_jobs.values())
        
        return {
            'total_jobs': total_jobs,
            'active_jobs': len(self.active_jobs),
            'completed_jobs': completed_count,
            'failed_jobs': failed_count,
            'cancelled_jobs': cancelled_count,
            'total_videos_processed': total_videos_processed,
            'total_videos_failed': total_videos_failed,
            'queue_size': self.job_queue.qsize(),
            'max_concurrent_jobs': self.max_concurrent_jobs
        }

# Global batch processor instance
batch_processor = BatchProcessor()

# Batch processing helpers
def create_batch_from_pending_videos(limit: int = 10) -> Optional[str]:
    """Create a batch job from pending videos"""
    from app import Video  # Import here to avoid circular imports
    
    try:
        # Find videos that are uploaded but not processed
        pending_videos = Video.query.filter_by(status='uploaded').limit(limit).all()
        
        if not pending_videos:
            return None
        
        video_ids = [video.id for video in pending_videos]
        return batch_processor.create_batch_job(video_ids)
        
    except Exception as e:
        logger.error(f"Error creating batch from pending videos: {e}")
        return None

def process_all_failed_videos() -> Optional[str]:
    """Create a batch job to retry all failed videos"""
    from app import Video  # Import here to avoid circular imports
    
    try:
        failed_videos = Video.query.filter_by(status='failed').all()
        
        if not failed_videos:
            return None
        
        video_ids = [video.id for video in failed_videos]
        return batch_processor.create_batch_job(video_ids)
        
    except Exception as e:
        logger.error(f"Error creating batch for failed videos: {e}")
        return None

# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test batch processor
    processor = BatchProcessor(max_concurrent_jobs=1)
    
    # Create a test batch job
    test_video_ids = ["test1", "test2", "test3"]
    
    def progress_callback(data):
        print(f"Progress: {data['progress_percent']}% - Processing {data['current_video_id']}")
    
    job_id = processor.create_batch_job(test_video_ids, progress_callback)
    print(f"Created batch job: {job_id}")
    
    # Monitor job status
    import time
    while True:
        status = processor.get_job_status(job_id)
        if not status:
            break
            
        print(f"Job status: {status['status']} - {status['progress_percent']}%")
        
        if status['status'] in ['completed', 'failed', 'cancelled']:
            break
        
        time.sleep(2)
    
    # Get statistics
    stats = processor.get_statistics()
    print(f"Batch processor stats: {stats}")
    
    # Shutdown
    processor.shutdown()