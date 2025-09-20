# Sekai Multi-Character Memory System

A sophisticated memory system for narrative-based AI applications with multiple characters, implementing MIRIX + G-Memory + MemGPT architecture patterns.

## 🎯 Overview

This system addresses the unique challenges of multi-character narrative memory:

- **Character-to-User (C2U) Memory**: Each character maintains separate memories with the same user
- **Inter-Character (IC) Memory**: Characters remember interactions with each other 
- **World Memory (WM)**: Characters retain memories about evolving world state

## 🏗️ Architecture

### Core Components

1. **MIRIX-Inspired Memory Types**:
   - `Core`: Essential character traits and relationships
   - `Episodic`: Specific events and interactions  
   - `Semantic`: General knowledge and facts
   - `Procedural`: How-to knowledge and behavioral patterns
   - `Resource`: External references and information
   - `Knowledge Vault`: World state and environmental information

2. **G-Memory Graph Structures**:
   - `Insight Graph`: High-level patterns and insights
   - `Query Graph`: Query relationships and dependencies
   - `Interaction Graph`: Direct character interactions

3. **MemGPT-Style Hierarchical Storage**:
   - `Core Memory`: Always accessible essential information
   - `Recall Memory`: Recent interactions (last 20)
   - `Archival Memory`: Long-term compressed storage

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/BryanTJJ99/multiperspective_narrative_memory_system.git
cd multiperspective_narrative_memory_system

# Install dependencies
pip install -r requirements.txt

# Or install via setup.py
pip install -e .
```

### Basic Usage

```python
from sekai_memory_system import SekaiMemorySystem, MemoryType, RelationshipType
from data_loader import MemoryDataProcessor

# Initialize the memory system
memory_system = SekaiMemorySystem()

# Load your chapter data
processor = MemoryDataProcessor(memory_system)
stats = processor.process_json_file("memory_data.json")

print(f"Loaded {stats['memories_created']} memories across {stats['chapters_processed']} chapters")
```

### Adding Custom Memories

```python
# Add a memory manually
memory_id = memory_system.add_memory(
    character_perspective="Byleth",
    memory_type=MemoryType.EPISODIC,
    relationship_type=RelationshipType.INTER_CHARACTER,
    content="Had a strategic conversation with Dimitri about the upcoming project",
    chapter=15,
    participants=["Byleth", "Dimitri"],
    emotional_weight=0.3,
    importance=0.7,
    is_secret=False
)
```

### Retrieving Memories

```python
# Query memories
memories = memory_system.retrieve_memories(
    query="Byleth Dimitri relationship", 
    character_perspective="Byleth",
    top_k=5
)

for memory in memories:
    print(f"Chapter {memory.chapter}: {memory.content}")
```

### Character Relationship Analysis

```python
# Get relationship timeline
timeline = memory_system.get_character_relationship_timeline("Byleth", "Dimitri")

for memory in timeline:
    print(f"Chapter {memory.chapter}: {memory.emotional_weight:.2f} - {memory.content[:100]}...")
```

## 📊 Evaluation Pipeline

The system includes a comprehensive evaluation framework addressing three key questions:

### Question A: Retrieval Accuracy
```python
from evaluation_pipeline import ComprehensiveEvaluationPipeline

evaluator = ComprehensiveEvaluationPipeline(memory_system)
results = evaluator.run_full_evaluation()

print(f"Precision: {results['question_a_retrieval']['average_precision']:.3f}")
print(f"Recall: {results['question_a_retrieval']['average_recall']:.3f}")
print(f"F1 Score: {results['question_a_retrieval']['average_f1']:.3f}")
```

### Question B: Memory Consistency
```python
# Check consistency across characters and time
consistency = memory_system.analyze_memory_consistency()
print(f"Contradictions found: {len(consistency['contradictions'])}")
print(f"Character consistency scores: {consistency['character_consistency']}")
```

### Question C: Additional Metrics
- Character distinctiveness (personality consistency)
- Secret management (information asymmetry)
- Relationship tracking (evolution over time)

### Generate Evaluation Report

```python
# Generate comprehensive HTML report
evaluator.generate_evaluation_report(results, "my_evaluation_report.html")
```

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

## 📁 Project Structure

```
multiperspective_narrative_memory_system/
├── sekai_memory_system.py      # Core memory system implementation
├── data_loader.py              # Chapter data processing and memory extraction
├── evaluation_pipeline.py      # Comprehensive evaluation framework
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── README.md                   # This file
├── examples/
│   ├── basic_usage.py         # Basic usage examples
│   ├── advanced_queries.py    # Advanced memory queries
│   └── custom_evaluation.py   # Custom evaluation metrics
└── tests/
    ├── test_memory_system.py  # Unit tests for memory system
    ├── test_consistency.py    # Consistency testing
    └── test_evaluation.py     # Evaluation pipeline tests
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

## 💡 Advanced Features

### Memory Contradiction Detection

```python
# Detect contradictory memories
memory_id = "some-memory-id"
contradictions = memory_system.detect_contradictions(memory_id)

if contradictions:
    print(f"Found {len(contradictions)} potential contradictions")
    for contradiction_id in contradictions:
        conflicting_memory = memory_system.memories[contradiction_id]
        print(f"  Conflicts with: {conflicting_memory.content[:100]}...")
```

### Secret Management

```python
# Query only secret memories
secret_memories = [
    memory for memory in memory_system.memories.values() 
    if memory.is_secret
]

print(f"Total secrets in system: {len(secret_memories)}")

# Check secret compartmentalization
for memory in secret_memories:
    print(f"Secret known by: {memory.participants}")
```

### Graph Analysis

```python
import networkx as nx

# Analyze character interaction patterns
interaction_graph = memory_system.interaction_graph

# Find most connected characters
centrality = nx.degree_centrality(interaction_graph)
most_connected = max(centrality.items(), key=lambda x: x[1])
print(f"Most connected character: {most_connected[0]} (centrality: {most_connected[1]:.3f})")

# Find character clusters
clusters = list(nx.connected_components(interaction_graph.to_undirected()))
print(f"Character clusters: {clusters}")
```

### Memory Compression

```python
# MemGPT-style memory management
print(f"Core memory size: {len(memory_system.core_memory)}")
print(f"Recall memory size: {len(memory_system.recall_memory)}")
print(f"Archival memory size: {len(memory_system.archival_memory)}")

# Manually trigger compression
memory_system._compress_archival_memory()
```

## 🧪 Testing and Validation

### Unit Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_memory_system.py -v
python -m pytest tests/test_consistency.py -v
python -m pytest tests/test_evaluation.py -v
```

### Custom Test Cases

```python
from evaluation_pipeline import TestCase

# Create custom test case
custom_test = TestCase(
    id="custom_test_1",
    query="Byleth manipulation strategy",
    expected_memory_ids=["memory-123", "memory-456"],
    expected_characters=["Byleth"],
    chapter_range=(10, 20),
    description="Test Byleth's strategic thinking"
)

# Run evaluation on custom test
retrieval_evaluator = MemoryRetrievalEvaluator(memory_system)
results = retrieval_evaluator.evaluate_retrieval_accuracy([custom_test])
```

### Online Monitoring

```python
# Set up real-time evaluation
new_memory_ids = ["newly-added-memory-1", "newly-added-memory-2"]
online_metrics = evaluator.run_online_evaluation(new_memory_ids)

print(f"Immediate consistency: {online_metrics['immediate_consistency']:.3f}")
print(f"Secret security: {online_metrics['secret_security']:.3f}")
```

## 📈 Performance Optimization

### Memory Retrieval Optimization

```python
# Adjust retrieval parameters for performance
memory_system.retrieve_memories(
    query="character interaction",
    top_k=10,  # Reduce for faster queries
    character_perspective="Byleth"  # Filter by character for efficiency
)
```

### Embedding Model Selection

```python
# Use different embedding models for different performance/accuracy trade-offs
from sentence_transformers import SentenceTransformer

# Faster, smaller model
fast_embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Default

# More accurate, larger model  
accurate_embedder = SentenceTransformer('all-mpnet-base-v2')

# Initialize system with custom embedder
memory_system.embedder = accurate_embedder
```

### Database Optimization

```python
# Enable SQLite optimizations
memory_system._execute_sql("PRAGMA journal_mode=WAL")
memory_system._execute_sql("PRAGMA synchronous=NORMAL") 
memory_system._execute_sql("PRAGMA cache_size=10000")
```

## 🔍 Debugging and Monitoring

### Memory System Inspection

```python
# Inspect system state
print(f"Total memories: {len(memory_system.memories)}")
print(f"Characters tracked: {list(memory_system.core_memory.keys())}")

# Memory distribution by type
from collections import Counter
memory_types = Counter(m.memory_type for m in memory_system.memories.values())
print(f"Memory type distribution: {dict(memory_types)}")

# Relationship type distribution
rel_types = Counter(m.relationship_type for m in memory_system.memories.values())
print(f"Relationship type distribution: {dict(rel_types)}")
```

### Character Analysis

```python
# Analyze specific character
character = "Byleth"
char_memories = [m for m in memory_system.memories.values() if m.character_perspective == character]

print(f"\n{character} Analysis:")
print(f"  Total memories: {len(char_memories)}")
print(f"  Secrets: {len([m for m in char_memories if m.is_secret])}")
print(f"  Average emotional weight: {np.mean([m.emotional_weight for m in char_memories]):.3f}")
print(f"  Average importance: {np.mean([m.importance for m in char_memories]):.3f}")
```

### Visualization

```python
import matplotlib.pyplot as plt

# Plot memory timeline
chapters = [m.chapter for m in memory_system.memories.values()]
emotional_weights = [m.emotional_weight for m in memory_system.memories.values()]

plt.figure(figsize=(12, 6))
plt.scatter(chapters, emotional_weights, alpha=0.6)
plt.xlabel("Chapter")
plt.ylabel("Emotional Weight")
plt.title("Memory Emotional Weight Over Time")
plt.show()

# Character interaction heatmap
import seaborn as sns
import pandas as pd

# Create interaction matrix
characters = list(memory_system.core_memory.keys())
interaction_matrix = np.zeros((len(characters), len(characters)))

for i, char1 in enumerate(characters):
    for j, char2 in enumerate(characters):
        if char1 != char2:
            shared_memories = len([
                m for m in memory_system.memories.values()
                if char1 in m.participants and char2 in m.participants
            ])
            interaction_matrix[i][j] = shared_memories

plt.figure(figsize=(10, 8))
sns.heatmap(interaction_matrix, 
            xticklabels=characters, 
            yticklabels=characters,
            annot=True, 
            fmt='.0f',
            cmap='Blues')
plt.title("Character Interaction Frequency Matrix")
plt.show()
```

## 🛠️ Extending the System

### Custom Memory Types

```python
from enum import Enum

class CustomMemoryType(Enum):
    EMOTIONAL = "emotional"      # Emotional reactions and feelings
    STRATEGIC = "strategic"      # Long-term planning and goals
    SOCIAL = "social"           # Social dynamics and relationships

# Extend the system to use custom types
# (Requires modification to the core system)
```

### Custom Evaluation Metrics

```python
from evaluation_pipeline import EvaluationResult

def custom_metric_evaluator(memory_system):
    """Example custom evaluation metric"""
    
    # Calculate metric (e.g., dialogue coherence)
    coherence_score = 0.85  # Your calculation here
    
    return EvaluationResult(
        metric_name="dialogue_coherence",
        score=coherence_score,
        details={
            "coherence_score": coherence_score,
            "measurement_method": "custom_algorithm"
        },
        timestamp=datetime.now()
    )

# Add to evaluation pipeline
evaluator.custom_evaluators.append(custom_metric_evaluator)
```

### Integration with LLMs

```python
# Example: Use OpenAI API for memory summarization
import openai

def llm_memory_summarizer(memories: List[Memory]) -> str:
    """Summarize memories using LLM"""
    
    memory_texts = [m.content for m in memories]
    combined_text = "\n".join(memory_texts)
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Summarize these character memories concisely."},
            {"role": "user", "content": combined_text}
        ]
    )
    
    return response.choices[0].message.content

# Use in memory compression
memory_system.llm_summarizer = llm_memory_summarizer
```

## 🚀 Production Deployment

### Docker Setup

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "api_server.py"]
```

### API Endpoint Example

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
memory_system = SekaiMemorySystem()

@app.route('/memories/search', methods=['POST'])
def search_memories():
    data = request.json
    
    memories = memory_system.retrieve_memories(
        query=data['query'],
        character_perspective=data.get('character'),
        top_k=data.get('top_k', 5)
    )
    
    return jsonify({
        'memories': [
            {
                'id': m.id,
                'content': m.content,
                'character': m.character_perspective,
                'chapter': m.chapter,
                'importance': m.importance
            } for m in memories
        ]
    })

@app.route('/memories/add', methods=['POST'])
def add_memory():
    data = request.json
    
    memory_id = memory_system.add_memory(
        character_perspective=data['character'],
        memory_type=MemoryType(data['type']),
        relationship_type=RelationshipType(data['relationship_type']),
        content=data['content'],
        chapter=data['chapter'],
        participants=data['participants'],
        emotional_weight=data.get('emotional_weight', 0.0),
        importance=data.get('importance', 0.5),
        is_secret=data.get('is_secret', False)
    )
    
    return jsonify({'memory_id': memory_id})

if __name__ == '__main__':
    app.run(debug=True)
```

## 📚 References

This implementation is based on research from:

- **MemGPT: Towards LLMs as Operating Systems** (Packer et al., 2023)
- **Generative Agents: Interactive Simulacra of Human Behavior** (Park et al., 2023)
- **MIRIX: Multi-Agent Memory System for LLM-Based Agents** (Wang & Chen, 2025)
- **G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems** (Zhang et al., 2025)

