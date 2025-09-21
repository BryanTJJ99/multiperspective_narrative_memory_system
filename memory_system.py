# memory_system.py
"""
Enhanced Memory System Import Helper
Provides clean imports for the narrative memory system with LLM integration
Dynamic system with LLM capabilities is the default, static system available if needed
"""

# Default: Enhanced Dynamic Memory System (recommended for all use cases)
from dynamic_memory_system import (
    DynamicMemorySystem as MemorySystem,  # Primary enhanced system
    MemoryDataProcessor,
    MemoryUpdate,
    CharacterProfile,
    ContradictionResolver,
    RelationshipTracker,
    NewCharacterIntegrator,
    MemoryUpdateAPI,
    create_flask_app,
    LLMConfig  # NEW: LLM configuration
)

# Evaluation Pipeline (works with both systems)
from evaluation_pipeline import (
    ComprehensiveEvaluationPipeline,
    MemoryRetrievalEvaluator,
    MemoryConsistencyEvaluator,
    AdditionalMetricsEvaluator,
    EvaluationResult,
    TestCase
)

# Core memory types and enums
from sekai_memory_system import (
    Memory,
    MemoryType,
    RelationshipType,
    MemoryPerspective
)

# Static system (if needed for specific use cases)
from sekai_memory_system import SekaiMemorySystem as StaticMemorySystem

# Configuration
from config import DEFAULT_CONFIG, MemoryConfig

# Enhanced usage examples:
"""
# Standard usage with LLM enhancement (recommended)
from memory_system import MemorySystem, MemoryDataProcessor, ComprehensiveEvaluationPipeline

memory_system = MemorySystem(
    openai_api_key="your-openai-api-key",  # Enable LLM features
    llm_model="gpt-4"  # or "gpt-3.5-turbo" for cost savings
)
processor = MemoryDataProcessor(memory_system, openai_api_key="your-openai-api-key")
evaluator = ComprehensiveEvaluationPipeline(memory_system)

# Async usage for LLM features
import asyncio

async def process_narrative():
    stats = await processor.process_json_file("memory_data.json")
    result = await memory_system.add_new_chapter(new_chapter_data)
    return stats, result

# Without LLM (fallback mode)
memory_system = MemorySystem()  # No API key = statistical analysis only

# If you specifically need the static system
from memory_system import StaticMemorySystem
static_system = StaticMemorySystem()
"""

__version__ = "2.0.0"  # Updated for LLM integration
__all__ = [
    # Primary enhanced system
    'MemorySystem',
    'MemoryDataProcessor', 
    'ComprehensiveEvaluationPipeline',
    
    # Enhanced dynamic system components
    'MemoryUpdate',
    'CharacterProfile',
    'ContradictionResolver',
    'RelationshipTracker', 
    'NewCharacterIntegrator',
    'MemoryUpdateAPI',
    'create_flask_app',
    'LLMConfig',  # NEW
    
    # Evaluation components
    'MemoryRetrievalEvaluator',
    'MemoryConsistencyEvaluator', 
    'AdditionalMetricsEvaluator',
    'EvaluationResult',
    'TestCase',
    
    # Core types
    'Memory',
    'MemoryType',
    'RelationshipType',
    'MemoryPerspective',
    
    # Alternative systems
    'StaticMemorySystem',
    
    # Configuration
    'DEFAULT_CONFIG',
    'MemoryConfig'
]