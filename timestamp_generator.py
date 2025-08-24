import numpy as np
from collections import defaultdict
from decouple import config
import re
import math

class TimestampGenerator:
    def __init__(self):
        """Initialize timestamp generator with configuration"""
        self.min_timestamp_gap = config('MIN_TIMESTAMP_GAP', default=5.0, cast=float)
        self.max_topics = config('MAX_TOPICS', default=10, cast=int)
        self.confidence_threshold = config('CONFIDENCE_THRESHOLD', default=0.6, cast=float)
        
        # Timestamp generation strategies
        self.strategies = {
            'semantic_boundaries': self._generate_semantic_timestamps,
            'speaker_changes': self._generate_speaker_timestamps,
            'topic_transitions': self._generate_topic_timestamps,
            'engagement_points': self._generate_engagement_timestamps,
            'content_structure': self._generate_structural_timestamps
        }
    
    def generate_intelligent_timestamps(self, transcript_data, content_analysis):
        """
        Generate intelligent timestamps using multiple strategies
        
        Args:
            transcript_data (dict): Original transcript data from Deepgram
            content_analysis (dict): Content analysis results
            
        Returns:
            list: Optimized timestamp segments
        """
        try:
            # Generate timestamps using different strategies
            all_timestamps = []
            
            for strategy_name, strategy_func in self.strategies.items():
                strategy_timestamps = strategy_func(transcript_data, content_analysis)
                
                # Add strategy metadata
                for timestamp in strategy_timestamps:
                    timestamp['generation_strategy'] = strategy_name
                
                all_timestamps.extend(strategy_timestamps)
            
            # Merge and optimize timestamps
            optimized_timestamps = self._optimize_timestamps(all_timestamps, transcript_data)
            
            # Generate titles and descriptions
            final_timestamps = self._enhance_timestamps(optimized_timestamps, content_analysis)
            
            return final_timestamps
            
        except Exception as e:
            raise Exception(f"Timestamp generation failed: {str(e)}")
    
    def _generate_semantic_timestamps(self, transcript_data, content_analysis):
        """Generate timestamps based on semantic segment boundaries"""
        timestamps = []
        semantic_segments = content_analysis.get('semantic_segments', [])
        words = transcript_data.get('words', [])
        
        if not semantic_segments or not words:
            return timestamps
        
        try:
            for i, segment in enumerate(semantic_segments):
                # Find start time for this semantic segment
                start_sentence = segment.get('start_sentence', 0)
                end_sentence = segment.get('end_sentence', 0)
                
                # Estimate timestamps based on sentence positions
                start_time = self._estimate_time_from_sentence_position(start_sentence, len(transcript_data.get('paragraphs', [])), words)
                end_time = self._estimate_time_from_sentence_position(end_sentence + 1, len(transcript_data.get('paragraphs', [])), words)
                
                if end_time <= start_time:
                    end_time = start_time + 30.0  # Default 30-second segment
                
                timestamps.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'title': f"Segment {i + 1}",
                    'description': segment.get('text', '')[:100] + "..." if len(segment.get('text', '')) > 100 else segment.get('text', ''),
                    'confidence': segment.get('coherence_score', 0.7),
                    'segment_type': 'semantic_boundary'
                })
            
            return timestamps
            
        except Exception as e:
            print(f"Semantic timestamp generation error: {e}")
            return []
    
    def _generate_speaker_timestamps(self, transcript_data, content_analysis):
        """Generate timestamps based on speaker changes"""
        timestamps = []
        utterances = transcript_data.get('utterances', [])
        
        if len(utterances) < 2:
            return timestamps
        
        try:
            current_speaker = utterances[0].get('speaker', 0)
            segment_start = utterances[0].get('start', 0.0)
            segment_text = utterances[0].get('text', '')
            
            for i in range(1, len(utterances)):
                utterance = utterances[i]
                speaker = utterance.get('speaker', 0)
                
                # If speaker changes, create timestamp
                if speaker != current_speaker:
                    end_time = utterance.get('start', 0.0)
                    
                    # Only create timestamp if segment is long enough
                    if end_time - segment_start >= self.min_timestamp_gap:
                        timestamps.append({
                            'start_time': segment_start,
                            'end_time': end_time,
                            'title': f"Speaker {current_speaker + 1}",
                            'description': self._summarize_text(segment_text, 100),
                            'confidence': 0.8,
                            'segment_type': 'speaker_change'
                        })
                    
                    # Start new segment
                    current_speaker = speaker
                    segment_start = utterance.get('start', 0.0)
                    segment_text = utterance.get('text', '')
                else:
                    # Continue current segment
                    segment_text += ' ' + utterance.get('text', '')
            
            # Add final segment
            if utterances:
                final_utterance = utterances[-1]
                timestamps.append({
                    'start_time': segment_start,
                    'end_time': final_utterance.get('end', segment_start + 30.0),
                    'title': f"Speaker {current_speaker + 1}",
                    'description': self._summarize_text(segment_text, 100),
                    'confidence': 0.8,
                    'segment_type': 'speaker_change'
                })
            
            return timestamps
            
        except Exception as e:
            print(f"Speaker timestamp generation error: {e}")
            return []
    
    def _generate_topic_timestamps(self, transcript_data, content_analysis):
        """Generate timestamps based on topic transitions"""
        timestamps = []
        topics = content_analysis.get('topics', [])
        content_transitions = content_analysis.get('content_transitions', [])
        
        if not topics and not content_transitions:
            return timestamps
        
        try:
            # Use content transitions if available
            if content_transitions:
                prev_timestamp = 0.0
                
                for i, transition in enumerate(content_transitions):
                    timestamp_val = transition.get('timestamp', 0.0)
                    
                    if timestamp_val - prev_timestamp >= self.min_timestamp_gap:
                        timestamps.append({
                            'start_time': prev_timestamp,
                            'end_time': timestamp_val,
                            'title': transition.get('previous_topic', f'Topic {i}'),
                            'description': f"Discussion about {transition.get('previous_topic', 'this topic')}",
                            'confidence': transition.get('confidence', 0.7),
                            'segment_type': 'topic_transition'
                        })
                    
                    prev_timestamp = timestamp_val
                
                # Add final segment
                total_duration = max([w.get('end', 0) for w in transcript_data.get('words', [])]) if transcript_data.get('words') else 300.0
                if total_duration > prev_timestamp:
                    final_transition = content_transitions[-1] if content_transitions else {}
                    timestamps.append({
                        'start_time': prev_timestamp,
                        'end_time': total_duration,
                        'title': final_transition.get('new_topic', 'Final Topic'),
                        'description': f"Discussion about {final_transition.get('new_topic', 'this topic')}",
                        'confidence': 0.7,
                        'segment_type': 'topic_transition'
                    })
            
            return timestamps
            
        except Exception as e:
            print(f"Topic timestamp generation error: {e}")
            return []
    
    def _generate_engagement_timestamps(self, transcript_data, content_analysis):
        """Generate timestamps based on engagement markers"""
        timestamps = []
        engagement_markers = content_analysis.get('engagement_markers', {})
        words = transcript_data.get('words', [])
        
        if not engagement_markers or not words:
            return timestamps
        
        try:
            # Find high-engagement points in the transcript
            text = transcript_data.get('transcript', '')
            
            # Look for engagement patterns
            engagement_patterns = [
                (r'\b(subscribe|like|comment|share)\b', 'Call to Action'),
                (r'\?[^?]*\?', 'Q&A Section'),
                (r'\b(now|next|first|second|finally)\b', 'Transition'),
                (r'\b(important|key|crucial|remember)\b', 'Key Point')
            ]
            
            for pattern, segment_type in engagement_patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                
                for match in matches:
                    # Find approximate timestamp for this text position
                    char_position = match.start()
                    estimated_time = self._estimate_time_from_char_position(char_position, text, words)
                    
                    timestamps.append({
                        'start_time': max(0.0, estimated_time - 10.0),
                        'end_time': estimated_time + 20.0,
                        'title': segment_type,
                        'description': match.group(0),
                        'confidence': 0.6,
                        'segment_type': 'engagement_point'
                    })
            
            return timestamps[:5]  # Limit to top 5 engagement points
            
        except Exception as e:
            print(f"Engagement timestamp generation error: {e}")
            return []
    
    def _generate_structural_timestamps(self, transcript_data, content_analysis):
        """Generate timestamps based on content structure"""
        timestamps = []
        paragraphs = transcript_data.get('paragraphs', [])
        
        if len(paragraphs) < 2:
            return timestamps
        
        try:
            # Group paragraphs into logical sections
            section_length = max(2, len(paragraphs) // 5)  # Aim for ~5 sections
            
            for i in range(0, len(paragraphs), section_length):
                section_paragraphs = paragraphs[i:i + section_length]
                
                if section_paragraphs:
                    start_time = section_paragraphs[0].get('start', 0.0)
                    end_time = section_paragraphs[-1].get('end', start_time + 60.0)
                    
                    # Combine text from section paragraphs
                    section_text = ' '.join([p.get('text', '') for p in section_paragraphs])
                    
                    timestamps.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'title': f"Section {i // section_length + 1}",
                        'description': self._summarize_text(section_text, 100),
                        'confidence': 0.7,
                        'segment_type': 'structural_section'
                    })
            
            return timestamps
            
        except Exception as e:
            print(f"Structural timestamp generation error: {e}")
            return []
    
    def _optimize_timestamps(self, all_timestamps, transcript_data):
        """Optimize and merge overlapping timestamps"""
        if not all_timestamps:
            return []
        
        try:
            # Sort timestamps by start time
            sorted_timestamps = sorted(all_timestamps, key=lambda x: x['start_time'])
            
            # Merge overlapping timestamps
            merged = []
            current = sorted_timestamps[0].copy()
            
            for timestamp in sorted_timestamps[1:]:
                # If timestamps overlap significantly, merge them
                if timestamp['start_time'] <= current['end_time'] + self.min_timestamp_gap:
                    # Merge timestamps - keep the one with higher confidence
                    if timestamp['confidence'] > current['confidence']:
                        current['title'] = timestamp['title']
                        current['description'] = timestamp['description']
                        current['confidence'] = timestamp['confidence']
                        current['segment_type'] = timestamp['segment_type']
                    
                    current['end_time'] = max(current['end_time'], timestamp['end_time'])
                else:
                    merged.append(current)
                    current = timestamp.copy()
            
            merged.append(current)
            
            # Filter by confidence and ensure minimum gap
            filtered = []
            for timestamp in merged:
                if (timestamp['confidence'] >= self.confidence_threshold and
                    timestamp['end_time'] - timestamp['start_time'] >= self.min_timestamp_gap):
                    filtered.append(timestamp)
            
            # Limit to reasonable number of timestamps
            if len(filtered) > self.max_topics:
                filtered = sorted(filtered, key=lambda x: x['confidence'], reverse=True)[:self.max_topics]
                filtered = sorted(filtered, key=lambda x: x['start_time'])
            
            return filtered
            
        except Exception as e:
            print(f"Timestamp optimization error: {e}")
            return all_timestamps[:self.max_topics]
    
    def _enhance_timestamps(self, timestamps, content_analysis):
        """Enhance timestamps with better titles and descriptions"""
        try:
            key_phrases = content_analysis.get('key_phrases', {}).get('top_phrases', [])
            topics = content_analysis.get('topics', [])
            
            enhanced = []
            
            for i, timestamp in enumerate(timestamps):
                enhanced_timestamp = timestamp.copy()
                
                # Improve title based on content
                if timestamp.get('segment_type') == 'topic_transition' and topics:
                    # Use topic keywords for title
                    relevant_topic = self._find_relevant_topic(timestamp, topics)
                    if relevant_topic:
                        top_keywords = relevant_topic.get('keywords', [])[:3]
                        if top_keywords:
                            enhanced_timestamp['title'] = ' | '.join(top_keywords).title()
                
                elif timestamp.get('segment_type') == 'semantic_boundary':
                    # Use key phrases for semantic segments
                    if key_phrases and i < len(key_phrases):
                        phrase, count = key_phrases[i % len(key_phrases)]
                        enhanced_timestamp['title'] = phrase.title()
                
                # Ensure title is not too long
                if len(enhanced_timestamp['title']) > 50:
                    enhanced_timestamp['title'] = enhanced_timestamp['title'][:47] + "..."
                
                # Format time for display
                enhanced_timestamp['formatted_start'] = self._format_timestamp(timestamp['start_time'])
                enhanced_timestamp['formatted_end'] = self._format_timestamp(timestamp['end_time'])
                enhanced_timestamp['duration'] = timestamp['end_time'] - timestamp['start_time']
                
                enhanced.append(enhanced_timestamp)
            
            return enhanced
            
        except Exception as e:
            print(f"Timestamp enhancement error: {e}")
            return timestamps
    
    def _find_relevant_topic(self, timestamp, topics):
        """Find the most relevant topic for a timestamp"""
        # Simple heuristic: match by position or keywords in description
        for topic in topics:
            topic_keywords = topic.get('keywords', [])
            description = timestamp.get('description', '').lower()
            
            # Check if any topic keywords appear in the description
            for keyword in topic_keywords[:3]:
                if keyword.lower() in description:
                    return topic
        
        return topics[0] if topics else None
    
    def _estimate_time_from_sentence_position(self, sentence_pos, total_sentences, words):
        """Estimate timestamp from sentence position"""
        if not words or total_sentences == 0:
            return sentence_pos * 10.0  # Default: 10 seconds per sentence
        
        # Get total duration from words
        total_duration = max([w.get('end', 0) for w in words]) if words else 300.0
        
        # Estimate time based on sentence position
        return (sentence_pos / total_sentences) * total_duration
    
    def _estimate_time_from_char_position(self, char_pos, text, words):
        """Estimate timestamp from character position in text"""
        if not words or not text:
            return 0.0
        
        # Find the word that contains this character position
        current_pos = 0
        for word in words:
            word_text = word.get('word', '')
            if current_pos + len(word_text) >= char_pos:
                return word.get('start', 0.0)
            current_pos += len(word_text) + 1  # +1 for space
        
        # If not found, return proportional estimate
        if text:
            ratio = char_pos / len(text)
            total_duration = max([w.get('end', 0) for w in words])
            return ratio * total_duration
        
        return 0.0
    
    def _format_timestamp(self, seconds):
        """Format seconds into MM:SS or HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def _summarize_text(self, text, max_length):
        """Create a brief summary of text"""
        if len(text) <= max_length:
            return text
        
        # Try to break at sentence boundary
        sentences = text.split('.')
        if sentences and len(sentences[0]) <= max_length - 1:
            return sentences[0] + '.'
        
        # Otherwise truncate
        return text[:max_length - 3] + "..."

# Example usage and testing
if __name__ == "__main__":
    try:
        generator = TimestampGenerator()
        print("[SUCCESS] TimestampGenerator initialized successfully!")
        
        # Test with sample data
        sample_transcript = {
            'transcript': "Welcome to this tutorial. First we'll cover basics. Then advanced topics. Finally, we'll wrap up.",
            'words': [
                {'word': 'Welcome', 'start': 0.0, 'end': 0.5},
                {'word': 'to', 'start': 0.5, 'end': 0.7},
                {'word': 'this', 'start': 0.7, 'end': 1.0},
                {'word': 'tutorial', 'start': 1.0, 'end': 1.8}
            ],
            'utterances': [
                {'speaker': 0, 'start': 0.0, 'end': 10.0, 'text': "Welcome to this tutorial."},
                {'speaker': 0, 'start': 10.0, 'end': 20.0, 'text': "First we'll cover basics."}
            ],
            'paragraphs': [
                {'start': 0.0, 'end': 10.0, 'text': "Welcome to this tutorial."},
                {'start': 10.0, 'end': 20.0, 'text': "First we'll cover basics."}
            ]
        }
        
        sample_analysis = {
            'topics': [{'keywords': ['tutorial', 'basics'], 'confidence': 0.8}],
            'semantic_segments': [{'start_sentence': 0, 'end_sentence': 1, 'text': 'Sample segment', 'coherence_score': 0.7}],
            'content_transitions': [],
            'engagement_markers': {'questions': 0, 'calls_to_action': 1},
            'key_phrases': {'top_phrases': [('machine learning', 3), ('tutorial', 2)]}
        }
        
        timestamps = generator.generate_intelligent_timestamps(sample_transcript, sample_analysis)
        print(f"Generated {len(timestamps)} intelligent timestamps!")
        
    except Exception as e:
        print(f"[ERROR] TimestampGenerator test failed: {e}")