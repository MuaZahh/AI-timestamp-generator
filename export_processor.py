import json
import csv
from datetime import datetime, timedelta
import io
import re

class ExportProcessor:
    def __init__(self):
        """Initialize export processor"""
        pass
    
    def export_youtube_chapters(self, timestamps, video_title="My Video"):
        """
        Export timestamps as YouTube chapter format
        
        Args:
            timestamps (list): List of timestamp dictionaries
            video_title (str): Title of the video
            
        Returns:
            str: YouTube chapter format
        """
        if not timestamps:
            return "No timestamps available for export."
        
        # Sort timestamps by start time
        sorted_timestamps = sorted(timestamps, key=lambda x: x['start_time'])
        
        chapters = []
        chapters.append(f"VIDEO: {video_title}\n")
        chapters.append("CHAPTERS:")
        chapters.append("")
        
        for i, ts in enumerate(sorted_timestamps):
            start_time = self._format_youtube_time(ts['start_time'])
            title = ts.get('title', f'Chapter {i+1}')
            
            # Clean title for YouTube (remove special chars except basic ones)
            clean_title = re.sub(r'[^\w\s\-\.,!?()]', '', title)
            
            chapters.append(f"{start_time} - {clean_title}")
        
        chapters.append("")
        chapters.append("Generated with AI Timestamp Generator")
        chapters.append("https://github.com/your-repo/ai-timestamp-generator")
        
        return "\n".join(chapters)
    
    def export_srt_subtitles(self, timestamps, transcript_data=None):
        """
        Export timestamps as SRT subtitle format
        
        Args:
            timestamps (list): List of timestamp dictionaries
            transcript_data (dict): Original transcript data with word-level timing
            
        Returns:
            str: SRT subtitle format
        """
        if not timestamps:
            return "No timestamps available for SRT export."
        
        # Sort timestamps by start time
        sorted_timestamps = sorted(timestamps, key=lambda x: x['start_time'])
        
        srt_lines = []
        
        for i, ts in enumerate(sorted_timestamps, 1):
            start_time = self._format_srt_time(ts['start_time'])
            end_time = self._format_srt_time(ts.get('end_time', ts['start_time'] + 30))
            
            # Use description as subtitle text, or title if no description
            subtitle_text = ts.get('description', ts.get('title', f'Segment {i}'))
            
            # Clean subtitle text (remove excessive whitespace, limit length)
            subtitle_text = re.sub(r'\s+', ' ', subtitle_text.strip())
            if len(subtitle_text) > 100:
                subtitle_text = subtitle_text[:97] + "..."
            
            srt_lines.extend([
                str(i),
                f"{start_time} --> {end_time}",
                subtitle_text,
                ""
            ])
        
        return "\n".join(srt_lines)
    
    def export_json(self, timestamps, video_metadata=None, include_metadata=True):
        """
        Export timestamps as JSON format for developers
        
        Args:
            timestamps (list): List of timestamp dictionaries
            video_metadata (dict): Video metadata
            include_metadata (bool): Whether to include metadata
            
        Returns:
            str: JSON format
        """
        export_data = {
            "format": "ai-timestamp-generator-v1",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "timestamps": []
        }
        
        if include_metadata and video_metadata:
            export_data["metadata"] = {
                "video_id": video_metadata.get('id'),
                "filename": video_metadata.get('filename'),
                "duration": video_metadata.get('duration'),
                "file_size": video_metadata.get('file_size'),
                "created_at": video_metadata.get('created_at'),
                "status": video_metadata.get('status')
            }
        
        # Sort timestamps by start time
        sorted_timestamps = sorted(timestamps, key=lambda x: x['start_time'])
        
        for ts in sorted_timestamps:
            export_data["timestamps"].append({
                "id": ts.get('id'),
                "start_time": ts['start_time'],
                "end_time": ts.get('end_time'),
                "duration": ts.get('end_time', ts['start_time'] + 30) - ts['start_time'],
                "title": ts.get('title'),
                "description": ts.get('description'),
                "confidence": ts.get('confidence'),
                "segment_type": ts.get('segment_type'),
                "formatted_start": self._format_youtube_time(ts['start_time']),
                "formatted_end": self._format_youtube_time(ts.get('end_time', ts['start_time'] + 30))
            })
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    def export_csv(self, timestamps):
        """
        Export timestamps as CSV format for data analysis
        
        Args:
            timestamps (list): List of timestamp dictionaries
            
        Returns:
            str: CSV format
        """
        if not timestamps:
            return "No timestamps available for CSV export."
        
        # Sort timestamps by start time
        sorted_timestamps = sorted(timestamps, key=lambda x: x['start_time'])
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Index',
            'Start Time (seconds)',
            'End Time (seconds)', 
            'Duration (seconds)',
            'Start Time (formatted)',
            'End Time (formatted)',
            'Title',
            'Description',
            'Confidence',
            'Segment Type'
        ])
        
        # Write data rows
        for i, ts in enumerate(sorted_timestamps, 1):
            end_time = ts.get('end_time', ts['start_time'] + 30)
            duration = end_time - ts['start_time']
            
            writer.writerow([
                i,
                ts['start_time'],
                end_time,
                duration,
                self._format_youtube_time(ts['start_time']),
                self._format_youtube_time(end_time),
                ts.get('title', ''),
                ts.get('description', ''),
                ts.get('confidence', ''),
                ts.get('segment_type', '')
            ])
        
        return output.getvalue()
    
    def export_vtt(self, timestamps):
        """
        Export timestamps as WebVTT format for web players
        
        Args:
            timestamps (list): List of timestamp dictionaries
            
        Returns:
            str: WebVTT format
        """
        if not timestamps:
            return "WEBVTT\n\nNo timestamps available."
        
        # Sort timestamps by start time
        sorted_timestamps = sorted(timestamps, key=lambda x: x['start_time'])
        
        vtt_lines = ["WEBVTT", ""]
        
        for i, ts in enumerate(sorted_timestamps, 1):
            start_time = self._format_vtt_time(ts['start_time'])
            end_time = self._format_vtt_time(ts.get('end_time', ts['start_time'] + 30))
            
            # Use description as caption text, or title if no description
            caption_text = ts.get('description', ts.get('title', f'Segment {i}'))
            
            # Clean caption text
            caption_text = re.sub(r'\s+', ' ', caption_text.strip())
            if len(caption_text) > 100:
                caption_text = caption_text[:97] + "..."
            
            vtt_lines.extend([
                str(i),
                f"{start_time} --> {end_time}",
                caption_text,
                ""
            ])
        
        return "\n".join(vtt_lines)
    
    def export_plaintext(self, timestamps):
        """
        Export timestamps as simple plain text format
        
        Args:
            timestamps (list): List of timestamp dictionaries
            
        Returns:
            str: Plain text format
        """
        if not timestamps:
            return "No timestamps available."
        
        # Sort timestamps by start time
        sorted_timestamps = sorted(timestamps, key=lambda x: x['start_time'])
        
        lines = []
        lines.append("AI Generated Timestamps")
        lines.append("=" * 50)
        lines.append("")
        
        for i, ts in enumerate(sorted_timestamps, 1):
            start_time = self._format_youtube_time(ts['start_time'])
            title = ts.get('title', f'Segment {i}')
            description = ts.get('description', '')
            
            lines.append(f"{i:2d}. {start_time} - {title}")
            if description and description != title:
                lines.append(f"    {description}")
            lines.append("")
        
        lines.append("Generated with AI Timestamp Generator")
        
        return "\n".join(lines)
    
    def _format_youtube_time(self, seconds):
        """Format seconds as YouTube time (M:SS or H:MM:SS)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def _format_srt_time(self, seconds):
        """Format seconds as SRT time (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def _format_vtt_time(self, seconds):
        """Format seconds as WebVTT time (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    
    def get_available_formats(self):
        """Get list of available export formats"""
        return {
            'youtube': {
                'name': 'YouTube Chapters',
                'description': 'Copy-paste ready format for YouTube video descriptions',
                'extension': 'txt',
                'mime_type': 'text/plain'
            },
            'srt': {
                'name': 'SRT Subtitles', 
                'description': 'SubRip subtitle format for video players',
                'extension': 'srt',
                'mime_type': 'text/plain'
            },
            'vtt': {
                'name': 'WebVTT Subtitles',
                'description': 'Web Video Text Tracks for HTML5 players',
                'extension': 'vtt',
                'mime_type': 'text/vtt'
            },
            'json': {
                'name': 'JSON Data',
                'description': 'Machine-readable format for developers',
                'extension': 'json',
                'mime_type': 'application/json'
            },
            'csv': {
                'name': 'CSV Spreadsheet',
                'description': 'Comma-separated values for data analysis',
                'extension': 'csv',
                'mime_type': 'text/csv'
            },
            'txt': {
                'name': 'Plain Text',
                'description': 'Simple text format for sharing',
                'extension': 'txt',
                'mime_type': 'text/plain'
            }
        }
    
    def export_format(self, format_type, timestamps, **kwargs):
        """
        Export timestamps in the specified format
        
        Args:
            format_type (str): Export format ('youtube', 'srt', 'json', etc.)
            timestamps (list): List of timestamp dictionaries
            **kwargs: Additional arguments for specific formats
            
        Returns:
            str: Exported content
        """
        format_methods = {
            'youtube': self.export_youtube_chapters,
            'srt': self.export_srt_subtitles,
            'vtt': self.export_vtt,
            'json': self.export_json,
            'csv': self.export_csv,
            'txt': self.export_plaintext
        }
        
        if format_type not in format_methods:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return format_methods[format_type](timestamps, **kwargs)

# Example usage and testing
if __name__ == "__main__":
    # Test data
    test_timestamps = [
        {
            'id': '1',
            'start_time': 0.0,
            'end_time': 30.0,
            'title': 'Introduction',
            'description': 'Welcome to the tutorial about machine learning',
            'confidence': 0.9,
            'segment_type': 'topic_transition'
        },
        {
            'id': '2', 
            'start_time': 30.0,
            'end_time': 90.0,
            'title': 'Getting Started',
            'description': 'First steps in understanding the basics',
            'confidence': 0.8,
            'segment_type': 'semantic_boundary'
        },
        {
            'id': '3',
            'start_time': 90.0,
            'end_time': 150.0,
            'title': 'Advanced Topics',
            'description': 'Deep dive into complex concepts and applications',
            'confidence': 0.85,
            'segment_type': 'topic_transition'
        }
    ]
    
    exporter = ExportProcessor()
    
    print("=== YouTube Chapters ===")
    print(exporter.export_youtube_chapters(test_timestamps, "AI Tutorial"))
    print("\n" + "="*50 + "\n")
    
    print("=== SRT Subtitles ===")
    print(exporter.export_srt_subtitles(test_timestamps))
    print("\n" + "="*50 + "\n")
    
    print("=== JSON Export ===")
    print(exporter.export_json(test_timestamps))
    print("\n" + "="*50 + "\n")
    
    print("=== Available Formats ===")
    for fmt, info in exporter.get_available_formats().items():
        print(f"{fmt}: {info['name']} - {info['description']}")