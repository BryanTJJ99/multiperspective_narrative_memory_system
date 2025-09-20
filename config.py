# config.py
"""Configuration settings for the Sekai Memory System"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class MemoryConfig:
    # Database settings
    db_path: str = "demo_sekai_memory.db"
    
    # Embedding model settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Memory hierarchy settings
    core_memory_limit: int = 50      # Max memories in core
    recall_memory_limit: int = 20    # Max memories in recall
    archival_threshold: int = 100    # When to compress archival
    
    # Retrieval settings
    default_top_k: int = 5
    similarity_threshold: float = 0.3
    importance_weight: float = 0.4
    recency_weight: float = 0.3
    
    # Character settings
    characters: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.characters is None:
            self.characters = {
                "Byleth": {
                    "personality_traits": ["manipulative", "strategic", "charming"],
                    "memory_bias": 0.1,  # Slight positive bias in memory encoding
                    "secret_threshold": 0.8  # How secretive they are
                },
                "Dimitri": {
                    "personality_traits": ["intense", "passionate", "loyal"],
                    "memory_bias": -0.1,  # Slight negative bias (pessimistic)
                    "secret_threshold": 0.3  # More open
                },
                "Sylvain": {
                    "personality_traits": ["charming", "flirtatious", "conflicted"],
                    "memory_bias": 0.0,   # Neutral
                    "secret_threshold": 0.6
                },
                "Annette": {
                    "personality_traits": ["trusting", "optimistic", "loyal"],
                    "memory_bias": 0.3,   # Very positive bias
                    "secret_threshold": 0.2  # Very open
                },
                "Dedue": {
                    "personality_traits": ["observant", "protective", "methodical"],
                    "memory_bias": -0.2,  # Cautious/negative bias
                    "secret_threshold": 0.9  # Very secretive
                }
            }

# Default configuration instance
DEFAULT_CONFIG = MemoryConfig()