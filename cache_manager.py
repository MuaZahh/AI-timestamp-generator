import os
import json
import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, cache_dir="cache", default_ttl=3600):
        """
        Initialize cache manager
        
        Args:
            cache_dir (str): Directory for cache files
            default_ttl (int): Default time-to-live in seconds (1 hour)
        """
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        
        # Create cache directories
        self.cache_dir.mkdir(exist_ok=True)
        (self.cache_dir / "transcripts").mkdir(exist_ok=True)
        (self.cache_dir / "analysis").mkdir(exist_ok=True)
        (self.cache_dir / "exports").mkdir(exist_ok=True)
        (self.cache_dir / "audio").mkdir(exist_ok=True)
        
        logger.info(f"Cache manager initialized with directory: {self.cache_dir}")
    
    def _get_cache_key(self, prefix: str, identifier: str, params: Dict = None) -> str:
        """Generate cache key from identifier and parameters"""
        if params:
            params_str = json.dumps(params, sort_keys=True)
            identifier += f"_{params_str}"
        
        # Create hash of identifier for consistent filename
        hash_key = hashlib.md5(identifier.encode()).hexdigest()
        return f"{prefix}_{hash_key}"
    
    def _get_cache_path(self, cache_type: str, cache_key: str) -> Path:
        """Get full cache file path"""
        return self.cache_dir / cache_type / f"{cache_key}.cache"
    
    def _is_cache_valid(self, cache_path: Path, ttl: int) -> bool:
        """Check if cache file is still valid"""
        if not cache_path.exists():
            return False
        
        # Check if cache has expired
        file_age = time.time() - cache_path.stat().st_mtime
        return file_age < ttl
    
    def get(self, cache_type: str, identifier: str, params: Dict = None, ttl: int = None) -> Optional[Any]:
        """
        Get item from cache
        
        Args:
            cache_type (str): Type of cache ('transcripts', 'analysis', 'exports', 'audio')
            identifier (str): Unique identifier for the cached item
            params (dict): Additional parameters that affect the cache key
            ttl (int): Time-to-live override
            
        Returns:
            Cached data or None if not found/expired
        """
        try:
            if ttl is None:
                ttl = self.default_ttl
                
            cache_key = self._get_cache_key(cache_type, identifier, params)
            cache_path = self._get_cache_path(cache_type, cache_key)
            
            if not self._is_cache_valid(cache_path, ttl):
                return None
            
            # Load cached data
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            logger.debug(f"Cache hit: {cache_type}/{cache_key}")
            return cached_data
            
        except Exception as e:
            logger.warning(f"Cache read error for {cache_type}/{identifier}: {e}")
            return None
    
    def set(self, cache_type: str, identifier: str, data: Any, params: Dict = None, ttl: int = None):
        """
        Store item in cache
        
        Args:
            cache_type (str): Type of cache ('transcripts', 'analysis', 'exports', 'audio')
            identifier (str): Unique identifier for the cached item
            data: Data to cache
            params (dict): Additional parameters that affect the cache key
            ttl (int): Time-to-live override (not used for storage but for reference)
        """
        try:
            cache_key = self._get_cache_key(cache_type, identifier, params)
            cache_path = self._get_cache_path(cache_type, cache_key)
            
            # Ensure directory exists
            cache_path.parent.mkdir(exist_ok=True)
            
            # Store data with metadata
            cache_data = {
                'data': data,
                'cached_at': time.time(),
                'ttl': ttl or self.default_ttl,
                'identifier': identifier,
                'params': params
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data['data'], f)
            
            logger.debug(f"Cache stored: {cache_type}/{cache_key}")
            
        except Exception as e:
            logger.error(f"Cache write error for {cache_type}/{identifier}: {e}")
    
    def delete(self, cache_type: str, identifier: str, params: Dict = None):
        """Delete item from cache"""
        try:
            cache_key = self._get_cache_key(cache_type, identifier, params)
            cache_path = self._get_cache_path(cache_type, cache_key)
            
            if cache_path.exists():
                cache_path.unlink()
                logger.debug(f"Cache deleted: {cache_type}/{cache_key}")
                
        except Exception as e:
            logger.error(f"Cache delete error for {cache_type}/{identifier}: {e}")
    
    def clear_expired(self):
        """Remove all expired cache files"""
        cleared_count = 0
        
        try:
            for cache_type in ["transcripts", "analysis", "exports", "audio"]:
                cache_type_dir = self.cache_dir / cache_type
                if not cache_type_dir.exists():
                    continue
                
                for cache_file in cache_type_dir.glob("*.cache"):
                    try:
                        # Check if file is expired
                        with open(cache_file, 'rb') as f:
                            cached_data = pickle.load(f)
                        
                        # For backwards compatibility, assume default TTL if not stored
                        file_age = time.time() - cache_file.stat().st_mtime
                        if file_age > self.default_ttl:
                            cache_file.unlink()
                            cleared_count += 1
                            
                    except Exception as e:
                        logger.warning(f"Error checking cache file {cache_file}: {e}")
                        # Remove corrupted cache files
                        try:
                            cache_file.unlink()
                            cleared_count += 1
                        except:
                            pass
            
            logger.info(f"Cleared {cleared_count} expired cache files")
            return cleared_count
            
        except Exception as e:
            logger.error(f"Error clearing expired cache: {e}")
            return 0
    
    def clear_all(self):
        """Clear all cache files"""
        cleared_count = 0
        
        try:
            for cache_type in ["transcripts", "analysis", "exports", "audio"]:
                cache_type_dir = self.cache_dir / cache_type
                if not cache_type_dir.exists():
                    continue
                
                for cache_file in cache_type_dir.glob("*.cache"):
                    try:
                        cache_file.unlink()
                        cleared_count += 1
                    except Exception as e:
                        logger.warning(f"Error deleting cache file {cache_file}: {e}")
            
            logger.info(f"Cleared all {cleared_count} cache files")
            return cleared_count
            
        except Exception as e:
            logger.error(f"Error clearing all cache: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        stats = {
            'total_files': 0,
            'total_size_bytes': 0,
            'by_type': {}
        }
        
        try:
            for cache_type in ["transcripts", "analysis", "exports", "audio"]:
                cache_type_dir = self.cache_dir / cache_type
                if not cache_type_dir.exists():
                    stats['by_type'][cache_type] = {'files': 0, 'size_bytes': 0}
                    continue
                
                type_files = 0
                type_size = 0
                
                for cache_file in cache_type_dir.glob("*.cache"):
                    try:
                        file_size = cache_file.stat().st_size
                        type_files += 1
                        type_size += file_size
                        
                    except Exception as e:
                        logger.warning(f"Error getting stats for {cache_file}: {e}")
                
                stats['by_type'][cache_type] = {
                    'files': type_files,
                    'size_bytes': type_size,
                    'size_mb': round(type_size / 1024 / 1024, 2)
                }
                
                stats['total_files'] += type_files
                stats['total_size_bytes'] += type_size
            
            stats['total_size_mb'] = round(stats['total_size_bytes'] / 1024 / 1024, 2)
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return stats

# Transcript-specific cache helpers
class TranscriptCache:
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.cache_type = "transcripts"
        self.ttl = 24 * 3600  # 24 hours for transcripts
    
    def get(self, video_id: str, audio_hash: str = None) -> Optional[Dict]:
        """Get cached transcript"""
        params = {'audio_hash': audio_hash} if audio_hash else None
        return self.cache_manager.get(self.cache_type, video_id, params, self.ttl)
    
    def set(self, video_id: str, transcript_data: Dict, audio_hash: str = None):
        """Cache transcript data"""
        params = {'audio_hash': audio_hash} if audio_hash else None
        self.cache_manager.set(self.cache_type, video_id, transcript_data, params, self.ttl)
    
    def delete(self, video_id: str):
        """Delete cached transcript"""
        self.cache_manager.delete(self.cache_type, video_id)

# Content analysis cache helpers  
class AnalysisCache:
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.cache_type = "analysis"
        self.ttl = 12 * 3600  # 12 hours for analysis
    
    def get(self, video_id: str, transcript_hash: str = None) -> Optional[Dict]:
        """Get cached content analysis"""
        params = {'transcript_hash': transcript_hash} if transcript_hash else None
        return self.cache_manager.get(self.cache_type, video_id, params, self.ttl)
    
    def set(self, video_id: str, analysis_data: Dict, transcript_hash: str = None):
        """Cache content analysis data"""
        params = {'transcript_hash': transcript_hash} if transcript_hash else None
        self.cache_manager.set(self.cache_type, video_id, analysis_data, params, self.ttl)

# Export cache helpers
class ExportCache:
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.cache_type = "exports"
        self.ttl = 6 * 3600  # 6 hours for exports
    
    def get(self, video_id: str, format_type: str, timestamps_hash: str = None) -> Optional[str]:
        """Get cached export"""
        params = {
            'format': format_type,
            'timestamps_hash': timestamps_hash
        }
        return self.cache_manager.get(self.cache_type, video_id, params, self.ttl)
    
    def set(self, video_id: str, format_type: str, export_content: str, timestamps_hash: str = None):
        """Cache export content"""
        params = {
            'format': format_type,
            'timestamps_hash': timestamps_hash
        }
        self.cache_manager.set(self.cache_type, video_id, export_content, params, self.ttl)

# Global cache instance
cache_manager = CacheManager()
transcript_cache = TranscriptCache(cache_manager)
analysis_cache = AnalysisCache(cache_manager)
export_cache = ExportCache(cache_manager)

# Cache maintenance function
def maintain_cache():
    """Periodic cache maintenance"""
    try:
        cleared = cache_manager.clear_expired()
        stats = cache_manager.get_cache_stats()
        
        logger.info(f"Cache maintenance: cleared {cleared} expired files")
        logger.info(f"Cache stats: {stats['total_files']} files, {stats['total_size_mb']} MB")
        
        # If cache is getting too large (>500MB), clear older files
        if stats['total_size_bytes'] > 500 * 1024 * 1024:
            logger.warning("Cache size exceeded 500MB, clearing all cache")
            cache_manager.clear_all()
            
        return stats
        
    except Exception as e:
        logger.error(f"Cache maintenance error: {e}")
        return None

if __name__ == "__main__":
    # Test the cache system
    logging.basicConfig(level=logging.DEBUG)
    
    # Test basic caching
    cache = CacheManager("test_cache")
    
    # Test storing and retrieving data
    test_data = {"message": "Hello, World!", "timestamp": time.time()}
    cache.set("test", "example_key", test_data)
    
    retrieved = cache.get("test", "example_key")
    print(f"Stored: {test_data}")
    print(f"Retrieved: {retrieved}")
    
    # Test cache expiration
    time.sleep(1)
    expired = cache.get("test", "example_key", ttl=1)  # 1 second TTL
    print(f"Expired cache: {expired}")
    
    # Test cache stats
    stats = cache.get_cache_stats()
    print(f"Cache stats: {stats}")
    
    # Cleanup
    cache.clear_all()
    print("Test completed successfully!")