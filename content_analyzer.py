try:
    import spacy
    import nltk
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import pandas as pd
    from textblob import TextBlob
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    
from collections import defaultdict
import re
from decouple import config

class ContentAnalyzer:
    def __init__(self):
        """Initialize content analyzer with NLP models"""
        self.advanced_nlp_available = ADVANCED_NLP_AVAILABLE
        
        if not self.advanced_nlp_available:
            print("Warning: Advanced NLP libraries not available. Using basic analysis only.")
            return
        
        try:
            # Load spaCy model
            self.nlp = spacy.load(config('SPACY_MODEL', default='en_core_web_sm'))
            
            # Configuration
            self.min_segment_length = config('MIN_SEGMENT_LENGTH', default=30, cast=int)
            self.max_segment_length = config('MAX_SEGMENT_LENGTH', default=300, cast=int)
            self.similarity_threshold = config('SIMILARITY_THRESHOLD', default=0.7, cast=float)
            self.confidence_threshold = config('CONFIDENCE_THRESHOLD', default=0.6, cast=float)
            
            # Initialize TF-IDF vectorizer
            self.tfidf = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
        except Exception as e:
            print(f"Warning: Failed to initialize advanced NLP features: {str(e)}")
            self.advanced_nlp_available = False
    
    def analyze_content(self, transcript_data):
        """
        Comprehensive content analysis of transcript
        
        Args:
            transcript_data (dict): Transcript data from Deepgram
            
        Returns:
            dict: Analysis results with topics, segments, and insights
        """
        try:
            # Extract text and basic info
            text = transcript_data.get('transcript', '')
            words = transcript_data.get('words', [])
            utterances = transcript_data.get('utterances', [])
            paragraphs = transcript_data.get('paragraphs', [])
            
            if not text.strip():
                raise ValueError("No transcript text available for analysis")
            
            if not self.advanced_nlp_available:
                # Basic analysis without advanced NLP
                return {
                    'text_stats': self._basic_text_stats(text),
                    'topics': [],
                    'semantic_segments': [],
                    'sentiment_analysis': {'overall_polarity': 0.0, 'overall_subjectivity': 0.0},
                    'key_phrases': {'top_phrases': [], 'named_entities': []},
                    'speaker_insights': self._analyze_speaker_patterns(utterances),
                    'content_transitions': [],
                    'engagement_markers': self._detect_engagement_markers(text)
                }
            
            # Perform comprehensive analysis
            analysis = {
                'text_stats': self._analyze_text_statistics(text),
                'topics': self._extract_topics(text),
                'semantic_segments': self._create_semantic_segments(text, words),
                'sentiment_analysis': self._analyze_sentiment(text),
                'key_phrases': self._extract_key_phrases(text),
                'speaker_insights': self._analyze_speaker_patterns(utterances),
                'content_transitions': self._detect_content_transitions(paragraphs),
                'engagement_markers': self._detect_engagement_markers(text)
            }
            
            return analysis
            
        except Exception as e:
            raise Exception(f"Content analysis failed: {str(e)}")
    
    def _basic_text_stats(self, text):
        """Basic text statistics without advanced NLP"""
        words = text.split()
        sentences = text.split('.')
        
        return {
            'total_words': len(words),
            'total_sentences': len(sentences),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'lexical_diversity': len(set(words)) / len(words) if words else 0,
            'named_entities': [],
            'pos_distribution': {}
        }
    
    def _analyze_text_statistics(self, text):
        """Analyze basic text statistics"""
        doc = self.nlp(text)
        
        return {
            'total_words': len([token for token in doc if not token.is_space]),
            'total_sentences': len(list(doc.sents)),
            'avg_sentence_length': np.mean([len(sent.text.split()) for sent in doc.sents]),
            'lexical_diversity': len(set([token.lemma_.lower() for token in doc if token.is_alpha])) / len([token for token in doc if token.is_alpha]) if len([token for token in doc if token.is_alpha]) > 0 else 0,
            'named_entities': [(ent.text, ent.label_) for ent in doc.ents],
            'pos_distribution': self._get_pos_distribution(doc)
        }
    
    def _get_pos_distribution(self, doc):
        """Get part-of-speech distribution"""
        pos_counts = defaultdict(int)
        for token in doc:
            if not token.is_space and not token.is_punct:
                pos_counts[token.pos_] += 1
        total = sum(pos_counts.values())
        return {pos: count/total for pos, count in pos_counts.items()} if total > 0 else {}
    
    def _extract_topics(self, text):
        """Extract topics using TF-IDF and clustering"""
        try:
            # Split text into sentences for topic analysis
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents if len(sent.text.strip()) > 20]
            
            if len(sentences) < 2:
                return []
            
            # Create TF-IDF vectors
            tfidf_matrix = self.tfidf.fit_transform(sentences)
            
            # Determine optimal number of clusters
            n_clusters = min(5, len(sentences) // 3, 10)
            if n_clusters < 2:
                n_clusters = 2
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            # Extract topics from clusters
            topics = []
            feature_names = self.tfidf.get_feature_names_out()
            
            for i in range(n_clusters):
                # Get cluster center
                center = kmeans.cluster_centers_[i]
                
                # Get top words for this cluster
                top_indices = center.argsort()[-10:][::-1]
                top_words = [feature_names[idx] for idx in top_indices if center[idx] > 0.1]
                
                # Get sentences in this cluster
                cluster_sentences = [sentences[j] for j, cluster in enumerate(clusters) if cluster == i]
                
                if top_words and cluster_sentences:
                    topics.append({
                        'id': i,
                        'keywords': top_words,
                        'sentences': cluster_sentences[:3],  # Top 3 representative sentences
                        'confidence': float(np.mean([center[idx] for idx in top_indices[:5]]))
                    })
            
            return sorted(topics, key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            print(f"Topic extraction error: {e}")
            return []
    
    def _create_semantic_segments(self, text, words):
        """Create semantic segments based on topic coherence"""
        try:
            doc = self.nlp(text)
            sentences = list(doc.sents)
            
            if len(sentences) < 2:
                return []
            
            # Calculate sentence similarities
            sentence_vectors = []
            for sent in sentences:
                if sent.vector_norm > 0:
                    sentence_vectors.append(sent.vector)
                else:
                    sentence_vectors.append(np.zeros(self.nlp.vocab.vectors_length))
            
            sentence_vectors = np.array(sentence_vectors)
            
            # Find semantic boundaries
            segments = []
            current_segment_start = 0
            
            for i in range(1, len(sentences)):
                # Calculate similarity between consecutive sentences
                similarity = cosine_similarity([sentence_vectors[i-1]], [sentence_vectors[i]])[0][0]
                
                # If similarity drops below threshold, start new segment
                if similarity < self.similarity_threshold or (i - current_segment_start) > 10:
                    segment_text = ' '.join([sent.text for sent in sentences[current_segment_start:i]])
                    
                    if len(segment_text.strip()) > self.min_segment_length:
                        segments.append({
                            'start_sentence': current_segment_start,
                            'end_sentence': i - 1,
                            'text': segment_text,
                            'length': len(segment_text),
                            'coherence_score': float(np.mean([cosine_similarity([sentence_vectors[j]], [sentence_vectors[j+1]])[0][0] 
                                                           for j in range(current_segment_start, i-1) if j+1 < len(sentence_vectors)]))
                        })
                    
                    current_segment_start = i
            
            # Add final segment
            if current_segment_start < len(sentences):
                segment_text = ' '.join([sent.text for sent in sentences[current_segment_start:]])
                if len(segment_text.strip()) > self.min_segment_length:
                    segments.append({
                        'start_sentence': current_segment_start,
                        'end_sentence': len(sentences) - 1,
                        'text': segment_text,
                        'length': len(segment_text),
                        'coherence_score': 0.8  # Default for final segment
                    })
            
            return segments
            
        except Exception as e:
            print(f"Semantic segmentation error: {e}")
            return []
    
    def _analyze_sentiment(self, text):
        """Analyze sentiment across the text"""
        try:
            blob = TextBlob(text)
            
            # Analyze sentence-level sentiment
            sentences = blob.sentences
            sentiment_scores = []
            
            for sentence in sentences:
                sentiment_scores.append({
                    'text': str(sentence),
                    'polarity': sentence.sentiment.polarity,
                    'subjectivity': sentence.sentiment.subjectivity
                })
            
            # Calculate overall sentiment
            avg_polarity = np.mean([s['polarity'] for s in sentiment_scores])
            avg_subjectivity = np.mean([s['subjectivity'] for s in sentiment_scores])
            
            return {
                'overall_polarity': float(avg_polarity),
                'overall_subjectivity': float(avg_subjectivity),
                'sentiment_trend': sentiment_scores,
                'emotional_intensity': float(np.std([s['polarity'] for s in sentiment_scores]))
            }
            
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return {'overall_polarity': 0.0, 'overall_subjectivity': 0.0, 'sentiment_trend': [], 'emotional_intensity': 0.0}
    
    def _extract_key_phrases(self, text):
        """Extract key phrases using spaCy"""
        try:
            doc = self.nlp(text)
            
            # Extract noun phrases
            noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks 
                          if len(chunk.text) > 3 and not chunk.root.is_stop]
            
            # Extract named entities
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            # Count frequencies
            phrase_counts = defaultdict(int)
            for phrase in noun_phrases:
                phrase_counts[phrase] += 1
            
            # Get most frequent phrases
            top_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            
            return {
                'top_phrases': top_phrases,
                'named_entities': entities,
                'technical_terms': self._identify_technical_terms(doc)
            }
            
        except Exception as e:
            print(f"Key phrase extraction error: {e}")
            return {'top_phrases': [], 'named_entities': [], 'technical_terms': []}
    
    def _identify_technical_terms(self, doc):
        """Identify technical terms and jargon"""
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+\d+\w*\b',  # Alphanumeric terms
            r'\b\w*[Tt]ech\w*\b',  # Tech-related words
            r'\b\w*[Aa]pi\w*\b'  # API-related terms
        ]
        
        technical_terms = []
        text = doc.text
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, text)
            technical_terms.extend(matches)
        
        return list(set(technical_terms))[:10]  # Return unique terms, limit to 10
    
    def _analyze_speaker_patterns(self, utterances):
        """Analyze speaker patterns and transitions"""
        if not utterances:
            return {}
        
        try:
            speaker_stats = defaultdict(lambda: {'duration': 0, 'word_count': 0, 'segments': 0})
            transitions = []
            
            for i, utterance in enumerate(utterances):
                speaker = utterance.get('speaker', 0)
                duration = utterance.get('end', 0) - utterance.get('start', 0)
                word_count = len(utterance.get('text', '').split())
                
                speaker_stats[speaker]['duration'] += duration
                speaker_stats[speaker]['word_count'] += word_count
                speaker_stats[speaker]['segments'] += 1
                
                # Track speaker transitions
                if i > 0 and utterances[i-1].get('speaker', 0) != speaker:
                    transitions.append({
                        'timestamp': utterance.get('start', 0),
                        'from_speaker': utterances[i-1].get('speaker', 0),
                        'to_speaker': speaker
                    })
            
            return {
                'speaker_statistics': dict(speaker_stats),
                'speaker_transitions': transitions,
                'dominant_speaker': max(speaker_stats.keys(), key=lambda x: speaker_stats[x]['duration']) if speaker_stats else 0
            }
            
        except Exception as e:
            print(f"Speaker analysis error: {e}")
            return {}
    
    def _detect_content_transitions(self, paragraphs):
        """Detect major content transitions"""
        if not paragraphs:
            return []
        
        try:
            transitions = []
            
            for i in range(1, len(paragraphs)):
                current_text = paragraphs[i].get('text', '')
                previous_text = paragraphs[i-1].get('text', '')
                
                if len(current_text) < 20 or len(previous_text) < 20:
                    continue
                
                # Analyze semantic similarity between paragraphs
                current_doc = self.nlp(current_text)
                previous_doc = self.nlp(previous_text)
                
                if current_doc.vector_norm > 0 and previous_doc.vector_norm > 0:
                    similarity = current_doc.similarity(previous_doc)
                    
                    # If similarity is low, it's likely a topic transition
                    if similarity < 0.5:
                        transitions.append({
                            'timestamp': paragraphs[i].get('start', 0),
                            'transition_type': 'topic_change',
                            'confidence': 1.0 - similarity,
                            'previous_topic': self._summarize_text(previous_text),
                            'new_topic': self._summarize_text(current_text)
                        })
            
            return transitions
            
        except Exception as e:
            print(f"Content transition detection error: {e}")
            return []
    
    def _detect_engagement_markers(self, text):
        """Detect markers that indicate engagement points"""
        markers = {
            'questions': len(re.findall(r'\?', text)),
            'exclamations': len(re.findall(r'!', text)),
            'calls_to_action': len(re.findall(r'\b(subscribe|like|comment|share|click|visit|check out|remember to)\b', text, re.IGNORECASE)),
            'transitions': len(re.findall(r'\b(now|next|first|second|finally|in conclusion|let\'s|so)\b', text, re.IGNORECASE)),
            'emphasis': len(re.findall(r'\b(very|really|absolutely|definitely|important|key|crucial)\b', text, re.IGNORECASE))
        }
        
        return markers
    
    def _summarize_text(self, text, max_length=50):
        """Create a brief summary of text"""
        if len(text) <= max_length:
            return text
        
        # Get first sentence or truncate
        sentences = text.split('.')
        if sentences and len(sentences[0]) <= max_length:
            return sentences[0] + '.'
        
        return text[:max_length] + '...'

# Example usage and testing
if __name__ == "__main__":
    try:
        analyzer = ContentAnalyzer()
        print("[SUCCESS] ContentAnalyzer initialized successfully!")
        
        # Test with sample text
        sample_transcript = {
            'transcript': "Welcome to this tutorial on machine learning. First, we'll discuss supervised learning algorithms. These algorithms learn from labeled data. Next, we'll explore unsupervised learning techniques. These methods find patterns in data without labels. Finally, we'll cover deep learning and neural networks.",
            'words': [],
            'utterances': [],
            'paragraphs': [
                {'start': 0.0, 'end': 10.0, 'text': "Welcome to this tutorial on machine learning."},
                {'start': 10.0, 'end': 20.0, 'text': "First, we'll discuss supervised learning algorithms."},
                {'start': 20.0, 'end': 35.0, 'text': "These algorithms learn from labeled data. Next, we'll explore unsupervised learning techniques."},
                {'start': 35.0, 'end': 50.0, 'text': "These methods find patterns in data without labels. Finally, we'll cover deep learning and neural networks."}
            ]
        }
        
        analysis = analyzer.analyze_content(sample_transcript)
        print(f"Analysis completed! Found {len(analysis['topics'])} topics and {len(analysis['semantic_segments'])} segments.")
        
    except Exception as e:
        print(f"[ERROR] ContentAnalyzer test failed: {e}")