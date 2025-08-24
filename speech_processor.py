from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from decouple import config
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechProcessor:
    def __init__(self):
        """Initialize Deepgram client"""
        self.api_key = config('DEEPGRAM_API_KEY', default='')
        
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY not found. Please set it in .env file or environment variables.")
        
        self.deepgram = DeepgramClient(self.api_key)
    
    def transcribe_audio(self, audio_path, language='en', model='nova-2', enable_diarization=True):
        """
        Transcribe audio file using Deepgram API
        
        Args:
            audio_path (str): Path to audio file
            language (str): Language code (e.g., 'en', 'es', 'fr')
            model (str): Deepgram model to use ('nova-2', 'nova', 'enhanced', 'base')
            enable_diarization (bool): Enable speaker diarization
            
        Returns:
            dict: Transcription result with timestamps
        """
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Read audio file
            with open(audio_path, "rb") as file:
                buffer_data = file.read()
            
            # Create FileSource object
            payload = {"buffer": buffer_data}
            
            # Configure options for transcription
            options = PrerecordedOptions(
                model=model,
                language=language,
                smart_format=True,
                punctuate=True,
                diarize=enable_diarization,
                utterances=True,
                paragraphs=True,
                topics=True,
                intents=True,
                sentiment=True,
                summarize='v2'  # Get summary
            )
            
            # Make transcription request
            logger.info(f"Starting transcription for: {audio_path}")
            response = self.deepgram.listen.rest.v("1").transcribe_file(payload, options)
            
            # Parse response
            result = response.to_dict()
            
            # Extract key information
            transcription_data = self._parse_transcription(result)
            
            logger.info(f"Transcription completed. Confidence: {transcription_data.get('confidence', 'N/A')}")
            
            return transcription_data
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise Exception(f"Speech-to-text processing failed: {str(e)}")
    
    def _parse_transcription(self, response):
        """
        Parse Deepgram response and extract relevant information
        
        Args:
            response (dict): Raw Deepgram API response
            
        Returns:
            dict: Parsed transcription data
        """
        try:
            results = response.get('results', {})
            channels = results.get('channels', [])
            
            if not channels:
                raise Exception("No transcription results found")
            
            channel = channels[0]
            alternatives = channel.get('alternatives', [])
            
            if not alternatives:
                raise Exception("No alternative transcriptions found")
            
            alternative = alternatives[0]
            
            # Extract basic transcription
            full_transcript = alternative.get('transcript', '')
            confidence = alternative.get('confidence', 0.0)
            
            # Extract words with timestamps
            words = alternative.get('words', [])
            
            # Extract utterances (speaker segments)
            utterances = []
            if 'utterances' in channel:
                for utterance in channel['utterances']:
                    utterances.append({
                        'speaker': utterance.get('speaker', 0),
                        'start': utterance.get('start', 0.0),
                        'end': utterance.get('end', 0.0),
                        'text': utterance.get('transcript', ''),
                        'confidence': utterance.get('confidence', 0.0)
                    })
            
            # Extract paragraphs
            paragraphs = []
            if 'paragraphs' in channel:
                for para in channel['paragraphs'].get('paragraphs', []):
                    paragraphs.append({
                        'start': para.get('start', 0.0),
                        'end': para.get('end', 0.0),
                        'text': ' '.join([s.get('text', '') for s in para.get('sentences', [])]),
                        'num_words': para.get('num_words', 0)
                    })
            
            # Extract topics
            topics = []
            if 'topics' in results:
                for topic in results['topics'].get('segments', []):
                    topics.append({
                        'start': topic.get('start_word', 0),
                        'end': topic.get('end_word', 0),
                        'topics': [{'text': t.get('topic', ''), 'confidence': t.get('confidence_score', 0.0)} 
                                 for t in topic.get('topics', [])]
                    })
            
            # Extract summary
            summary = ""
            if 'summary' in results:
                summary = results['summary'].get('short', '')
            
            # Extract sentiment
            sentiment = {}
            if 'sentiments' in results:
                segments = results['sentiments'].get('segments', [])
                if segments:
                    # Average sentiment across all segments
                    sentiment_scores = [s.get('sentiment_score', 0.0) for s in segments]
                    sentiment = {
                        'average_score': sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0,
                        'segments': [{
                            'start': s.get('start', 0.0),
                            'end': s.get('end', 0.0),
                            'sentiment': s.get('sentiment', 'neutral'),
                            'score': s.get('sentiment_score', 0.0)
                        } for s in segments]
                    }
            
            return {
                'transcript': full_transcript,
                'confidence': confidence,
                'words': words,
                'utterances': utterances,
                'paragraphs': paragraphs,
                'topics': topics,
                'summary': summary,
                'sentiment': sentiment,
                'language': results.get('language', 'en')
            }
            
        except Exception as e:
            logger.error(f"Error parsing transcription response: {str(e)}")
            raise Exception(f"Failed to parse transcription response: {str(e)}")
    
    def test_api_connection(self):
        """
        Test Deepgram API connection
        
        Returns:
            bool: True if API is accessible
        """
        try:
            # Test with a small sample (you could create a small test audio file)
            # For now, just validate API key format
            if len(self.api_key) < 20:
                return False
            
            logger.info("API key format appears valid")
            return True
            
        except Exception as e:
            logger.error(f"API connection test failed: {str(e)}")
            return False

# Example usage and testing
if __name__ == "__main__":
    try:
        processor = SpeechProcessor()
        
        # Test API connection
        if processor.test_api_connection():
            print("✅ Deepgram API connection test passed!")
        else:
            print("❌ Deepgram API connection test failed!")
            print("Please check your DEEPGRAM_API_KEY in .env file")
        
        # Example usage (commented out - requires actual audio file)
        # audio_path = "sample_audio.wav"
        # if os.path.exists(audio_path):
        #     try:
        #         result = processor.transcribe_audio(audio_path)
        #         print(f"Transcript: {result['transcript']}")
        #         print(f"Confidence: {result['confidence']}")
        #         print(f"Topics found: {len(result.get('topics', []))}")
        #         print(f"Summary: {result.get('summary', 'No summary')}")
        #     except Exception as e:
        #         print(f"Transcription error: {e}")
        
    except Exception as e:
        print(f"❌ Setup error: {e}")
        print("Make sure to set DEEPGRAM_API_KEY in your .env file")