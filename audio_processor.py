import ffmpeg
import os
import tempfile
import subprocess
from pathlib import Path
from decouple import config

class AudioProcessor:
    def __init__(self):
        # Get FFmpeg path from config or use system FFmpeg
        self.ffmpeg_path = config('FFMPEG_PATH', default='ffmpeg')
        
        # If local FFmpeg exists, use it
        local_ffmpeg = Path(__file__).parent / "ffmpeg" / "ffmpeg.exe"
        if local_ffmpeg.exists():
            self.ffmpeg_path = str(local_ffmpeg)
    
    def extract_audio(self, video_path, output_path=None, format='wav', sample_rate=16000):
        """
        Extract audio from video file using FFmpeg
        
        Args:
            video_path (str): Path to input video file
            output_path (str): Path for output audio file (optional)
            format (str): Output audio format ('wav', 'mp3', 'flac')
            sample_rate (int): Sample rate for output audio
            
        Returns:
            str: Path to extracted audio file
        """
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Generate output path if not provided
            if output_path is None:
                video_name = Path(video_path).stem
                temp_dir = tempfile.gettempdir()
                output_path = os.path.join(temp_dir, f"{video_name}_audio.{format}")
            
            # Build FFmpeg command
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le' if format == 'wav' else 'libmp3lame' if format == 'mp3' else 'flac',
                '-ar', str(sample_rate),
                '-ac', '1',  # Mono channel
                '-y',  # Overwrite output file
                output_path
            ]
            
            # Run FFmpeg command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"FFmpeg error: {result.stderr}")
            
            if not os.path.exists(output_path):
                raise Exception("Audio extraction failed: output file not created")
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Audio extraction failed: {str(e)}")
    
    def get_video_info(self, video_path):
        """
        Get information about video file (duration, format, etc.)
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            dict: Video information
        """
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-f', 'null',
                '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse duration from stderr
            duration = None
            for line in result.stderr.split('\n'):
                if 'Duration:' in line:
                    duration_str = line.split('Duration:')[1].split(',')[0].strip()
                    # Convert duration string (HH:MM:SS.ms) to seconds
                    time_parts = duration_str.split(':')
                    if len(time_parts) == 3:
                        hours = float(time_parts[0])
                        minutes = float(time_parts[1])
                        seconds = float(time_parts[2])
                        duration = hours * 3600 + minutes * 60 + seconds
                    break
            
            return {
                'duration': duration,
                'path': video_path,
                'size': os.path.getsize(video_path)
            }
            
        except Exception as e:
            raise Exception(f"Failed to get video info: {str(e)}")
    
    def cleanup_temp_audio(self, audio_path):
        """
        Clean up temporary audio files
        
        Args:
            audio_path (str): Path to audio file to delete
        """
        try:
            if os.path.exists(audio_path) and tempfile.gettempdir() in audio_path:
                os.remove(audio_path)
                return True
        except Exception:
            pass
        return False
    
    def test_ffmpeg(self):
        """
        Test if FFmpeg is working properly
        
        Returns:
            bool: True if FFmpeg is working
        """
        try:
            result = subprocess.run([self.ffmpeg_path, '-version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception:
            return False

# Example usage and testing
if __name__ == "__main__":
    processor = AudioProcessor()
    
    # Test FFmpeg
    if processor.test_ffmpeg():
        print("✅ FFmpeg is working correctly!")
    else:
        print("❌ FFmpeg test failed")
        print(f"FFmpeg path: {processor.ffmpeg_path}")
    
    # Example usage (commented out - requires actual video file)
    # video_path = "sample_video.mp4"
    # if os.path.exists(video_path):
    #     try:
    #         # Get video info
    #         info = processor.get_video_info(video_path)
    #         print(f"Video duration: {info['duration']} seconds")
    #         
    #         # Extract audio
    #         audio_path = processor.extract_audio(video_path)
    #         print(f"Audio extracted to: {audio_path}")
    #         
    #         # Clean up
    #         processor.cleanup_temp_audio(audio_path)
    #         
    #     except Exception as e:
    #         print(f"Error: {e}")