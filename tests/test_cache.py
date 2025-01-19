import pytest
from datetime import datetime, timedelta
from time import sleep
from agentgenius.cache import ToolCallCache, CacheEntry

@pytest.fixture
def cache():
    return ToolCallCache(max_size=3, ttl_minutes=1)

def test_basic_cache_operations(cache):
    # Test setting and getting a value
    cache.set("test_tool", (1, 2), {"x": 3}, "result1")
    result = cache.get("test_tool", (1, 2), {"x": 3})
    assert result == "result1"

    # Test getting non-existent value
    result = cache.get("nonexistent", (), {})
    assert result is None

def test_cache_key_consistency(cache):
    # Test that same arguments produce same key
    key1 = cache._make_key("tool", (1, 2), {"x": 3})
    key2 = cache._make_key("tool", (1, 2), {"x": 3})
    assert key1 == key2

    # Test that different argument orders in kwargs produce same key
    key1 = cache._make_key("tool", (), {"a": 1, "b": 2})
    key2 = cache._make_key("tool", (), {"b": 2, "a": 1})
    assert key1 == key2

def test_cache_size_limit(cache):
    # Fill cache to max size
    cache.set("tool1", (), {}, "result1")
    cache.set("tool2", (), {}, "result2")
    cache.set("tool3", (), {}, "result3")
    
    # Add one more item, should remove oldest
    cache.set("tool4", (), {}, "result4")
    
    # First item should be gone
    assert cache.get("tool1", (), {}) is None
    
    # Other items should still be there
    assert cache.get("tool2", (), {}) == "result2"
    assert cache.get("tool3", (), {}) == "result3"
    assert cache.get("tool4", (), {}) == "result4"

def test_cache_expiration(cache):
    # Set TTL to 1 second for faster testing
    cache.ttl_minutes = 1/60  # 1 second
    
    cache.set("test_tool", (), {}, "result")
    
    # Should be available immediately
    assert cache.get("test_tool", (), {}) == "result"
    
    # Wait for expiration
    sleep(1.1)
    
    # Should be expired now
    assert cache.get("test_tool", (), {}) is None

def test_edge_cases(cache):
    # Test with None value
    cache.set("tool", (), {}, None)
    assert cache.get("tool", (), {}) is None

    # Test with empty string
    cache.set("tool", (), {}, "")
    assert cache.get("tool", (), {}) == ""

    # Test with complex arguments
    complex_args = ((1, "2", [3, 4], {"5": 6}), {"a": [1, 2], "b": {"c": 3}})
    cache.set("tool", complex_args[0], complex_args[1], "result")
    assert cache.get("tool", complex_args[0], complex_args[1]) == "result"
