<!-- README.md -->
# Multiperspective Narrative Memory System

A sophisticated memory system for narrative-based AI applications with multiple characters, implementing MIRIX + G-Memory + MemGPT architecture patterns with dynamic growth capabilities.

## 🎯 Overview

This system addresses the unique challenges of multi-character narrative memory:

- **Character-to-User (C2U) Memory**: Each character maintains separate memories with the same user
- **Inter-Character (IC) Memory**: Characters remember interactions with each other 
- **World Memory (WM)**: Characters retain memories about evolving world state
- **Dynamic Growth**: Automatically integrates new chapters and characters as your narrative evolves

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/multiperspective_narrative_memory_system.git
cd multiperspective_narrative_memory_system

# Install dependencies
pip install -r requirements.txt

# Or install via setup.py
pip install -e .
```

### Basic Usage

```python
from memory_system import MemorySystem, MemoryDataProcessor, ComprehensiveEvaluationPipeline

# Initialize the dynamic memory system
memory_system = MemorySystem()

# Load your chapter data
processor = MemoryDataProcessor(memory_system)
stats = processor.process_json_file("memory_data.json")

print(f"Loaded {stats['memories_created']} memories across {stats['chapters_processed']} chapters")

# Run comprehensive evaluation
evaluator = ComprehensiveEvaluationPipeline(memory_system)
results = evaluator.run_full_evaluation()
print(f"Overall system score: {results['overall_score']:.3f}")
```

### Adding New Chapters Dynamically

```python
# Add new chapters as your story grows
new_chapter = {
    "chapter_number": 51,
    "synopsis": "A new character, Elena, arrives and immediately recognizes the complex relationships..."
}

result = memory_system.add_new_chapter(new_chapter)
print(f"New characters discovered: {result['new_characters_discovered']}")
print(f"Memories created: {result['new_memories_created']}")
```

## 🏗️ Architecture

### Core Components

1. **Dynamic Memory System**: Auto-growing memory with new character integration
2. **MIRIX-Inspired Memory Types**:
   - `Core`: Essential character traits and relationships
   - `Episodic`: Specific events and interactions  
   - `Semantic`: General knowledge and facts
   - `Procedural`: How-to knowledge and behavioral patterns
   - `Resource`: External references and information
   - `Knowledge Vault`: World state and environmental information

3. **G-Memory Graph Structures**:
   - `Insight Graph`: High-level patterns and insights
   - `Query Graph`: Query relationships and dependencies
   - `Interaction Graph`: Direct character interactions

4. **MemGPT-Style Hierarchical Storage**:
   - `Core Memory`: Always accessible essential information
   - `Recall Memory`: Recent interactions (last 20)
   - `Archival Memory`: Long-term compressed storage

## 📊 Evaluation Pipeline

The system provides automated evaluation answering three key questions:

### Question A: Retrieval Accuracy
```python
# Automatically tests precision/recall on multiple query types
results = evaluator.run_full_evaluation()
print(f"Precision: {results['question_a_retrieval']['average_precision']:.3f}")
print(f"Recall: {results['question_a_retrieval']['average_recall']:.3f}")
```

### Question B: Memory Consistency
- **Temporal Consistency**: Character evolution over time
- **Cross-Character Consistency**: Shared events remembered consistently  
- **World State Consistency**: Environmental changes reflected across characters

### Question C: Additional Metrics
- **Character Distinctiveness**: Unique personality maintenance
- **Secret Management**: Information asymmetry handling
- **Relationship Tracking**: Dynamic relationship evolution

## 🌟 Dynamic Features

### Automatic New Character Integration

```python
# System automatically detects and integrates new characters
new_chapter = {
    "chapter_number": 52,
    "synopsis": "Dr. Sarah Chen joins the team as the new AI ethics consultant..."
}

result = memory_system.add_new_chapter(new_chapter)
# ✅ Creates character profile for "Dr. Sarah Chen"
# ✅ Infers personality traits from interactions
# ✅ Establishes relationship networks
# ✅ Updates character evolution tracking
```

### Real-Time File Watching

```python
# Set up automatic chapter processing
memory_system = MemorySystem(watch_directory="./new_chapters")
memory_system.start_file_watcher(check_interval=30)

# Now just drop new chapter files and system auto-processes them!
```

### Character Evolution Analysis

```python
# Track how characters develop over time
evolution = memory_system.get_character_evolution("Byleth")
print(f"Total memories: {evolution['total_memories']}")
print(f"Relationships formed: {len(evolution['relationship_development'])}")
print(f"Emotional journey: {evolution['emotional_journey']}")
```

## 📁 Project Structure

```
multiperspective_narrative_memory_system/
├── memory_data.json                   # Your narrative data (you provide)
├── sekai_memory_system.py            # Base memory system
├── dynamic_memory_system.py          # Dynamic growth system (primary)
├── data_loader.py                    # Chapter data processing
├── evaluation_pipeline.py            # Comprehensive evaluation
├── memory_system.py                  # Clean import interface
├── complete_example.py              # Full demonstration
├── config.py                        # Configuration settings
├── requirements.txt                  # Dependencies
├── setup.py                         # Package setup
├── README.md                         # This file
└── dynamic_integration_guide.md     # Advanced integration guide
```

## 🎮 Working with Your Data

### Data Format

Your `memory_data.json` should follow this structure:

```json
[
  {
    "chapter_number": 1,
    "synopsis": "Byleth Eisner steps into Garreg Mach Corp, the air buzzing with first-day energy..."
  },
  {
    "chapter_number": 2, 
    "synopsis": "Amidst the cafeteria bustle, Byleth finds a quiet corner to observe..."
  }
]
```

### Data Processing Features

The system automatically:
- **Extracts character mentions** from synopses
- **Identifies emotional context** (positive/negative events)
- **Detects secret content** (affairs, hidden information)
- **Determines memory importance** based on content patterns
- **Creates character-specific perspectives** on shared events

## 🔧 Configuration

Customize the system behavior via `config.py`:

```python
from config import DEFAULT_CONFIG

# Modify memory limits
DEFAULT_CONFIG.core_memory_limit = 100
DEFAULT_CONFIG.recall_memory_limit = 50

# Adjust retrieval settings
DEFAULT_CONFIG.similarity_threshold = 0.4
DEFAULT_CONFIG.importance_weight = 0.5

# Configure character-specific settings
DEFAULT_CONFIG.characters["Byleth"]["secret_threshold"] = 0.9
```

## 🌐 API Integration

### REST API Setup

```python
from memory_system import create_flask_app

app = create_flask_app(memory_system)
app.run(host='0.0.0.0', port=5000)
```

### API Endpoints

- `POST /api/chapters` - Add new chapter
- `GET /api/characters/<name>/evolution` - Character evolution data
- `GET /api/stats/growth` - Narrative growth statistics  
- `POST /api/memories/search` - Search memories
- `GET /api/system/status` - System status

### Example API Usage

```bash
# Add new chapter
curl -X POST http://localhost:5000/api/chapters \
     -H "Content-Type: application/json" \
     -d '{"chapter_number": 51, "synopsis": "New chapter content..."}'

# Search memories
curl -X POST http://localhost:5000/api/memories/search \
     -H "Content-Type: application/json" \
     -d '{"query": "secret affair", "filters": {"character": "Byleth"}}'
```

## 📈 Performance & Scalability

### Memory Optimization

- **Automatic Consolidation**: Similar memories are merged for efficiency
- **Hierarchical Storage**: MemGPT-style core/recall/archival management
- **Intelligent Indexing**: Vector embeddings for fast similarity search
- **Database Persistence**: SQLite with optimized storage

### Production Ready

- **Error Handling**: Robust error recovery and logging
- **Concurrent Processing**: Thread-safe file watching
- **Memory Management**: Automatic cleanup and compression
- **API Rate Limiting**: Production-grade API endpoints

## 📚 Examples

### Basic Setup

```python
# Run the complete demonstration
python complete_example.py

# Choose from:
# 1. Full system demonstration with evaluation
# 2. Interactive query mode
# 3. Both demonstrations
```

### Advanced Integration

```python
# Production setup with file watching
from memory_system import MemorySystem

memory_system = MemorySystem(
    db_path="production_memory.db",
    watch_directory="./story_chapters"
)

# Start background processing
memory_system.start_file_watcher(check_interval=60)

# Your narrative files are now automatically processed!
```

## 🔍 Evaluation & Monitoring

### Generate Evaluation Reports

```python
evaluator = ComprehensiveEvaluationPipeline(memory_system)
results = evaluator.run_full_evaluation()
evaluator.generate_evaluation_report(results, "my_report.html")
```

###