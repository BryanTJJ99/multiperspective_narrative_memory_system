"""
Sekai Multi-Character Memory System
Based on MIRIX + G-Memory + MemGPT implementation path

This system implements:
1. MIRIX-inspired memory types (Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault)
2. G-Memory hierarchical graph structure (insight, query, interaction graphs)
3. MemGPT-style hierarchical memory management

Key Features:
- Character-to-User (C2U) memories
- Inter-Character (IC) memories  
- World Memory (WM)
- Perspective-dependent memory storage
- Temporal consistency tracking
- Automated evaluation pipeline
"""

import json
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """MIRIX-inspired memory types"""
    CORE = "core"                    # Essential character traits, relationships
    EPISODIC = "episodic"           # Specific events and interactions
    SEMANTIC = "semantic"           # General knowledge and facts
    PROCEDURAL = "procedural"       # How to do things, patterns
    RESOURCE = "resource"           # External information, references
    KNOWLEDGE_VAULT = "knowledge_vault"  # World state, environmental info

class RelationshipType(Enum):
    """Types of relationships between characters"""
    CHARACTER_TO_USER = "C2U"
    INTER_CHARACTER = "IC" 
    WORLD_MEMORY = "WM"

class MemoryPerspective(Enum):
    """Whose perspective the memory is from"""
    BYLETH = "Byleth"
    DIMITRI = "Dimitri"
    SYLVAIN = "Sylvain"
    ANNETTE = "Annette"
    DEDUE = "Dedue"
    MERCEDES = "Mercedes"
    FELIX = "Felix"
    ASHE = "Ashe"
    WORLD = "World"  # Objective world state

@dataclass
class Memory:
    """Individual memory record"""
    id: str
    character_perspective: str
    memory_type: MemoryType
    relationship_type: RelationshipType
    content: str
    chapter: int
    timestamp: datetime
    participants: List[str]  # Characters involved in this memory
    emotional_weight: float  # -1.0 to 1.0 (negative to positive)
    importance: float       # 0.0 to 1.0
    is_secret: bool
    contradicts: List[str]  # Memory IDs this contradicts
    supports: List[str]     # Memory IDs this supports
    embedding: Optional[np.ndarray] = None

class SekaiMemorySystem:
    """
    Main memory system implementing MIRIX + G-Memory + MemGPT architecture
    """
    
    def __init__(self, db_path: str = "sekai_memory.db"):
        self.db_path = db_path
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # G-Memory inspired graph structures
        self.insight_graph = nx.DiGraph()      # High-level patterns and insights
        self.query_graph = nx.DiGraph()        # Query relationships and dependencies  
        self.interaction_graph = nx.DiGraph()  # Direct character interactions
        
        # Memory storage
        self.memories: Dict[str, Memory] = {}
        self.character_perspectives: Dict[str, List[str]] = defaultdict(list)
        
        # MemGPT-style hierarchical storage
        self.core_memory: Dict[str, Dict] = {}     # Always accessible
        self.archival_memory: List[str] = []       # Long-term storage
        self.recall_memory: List[str] = []         # Recent interactions
        
        self._initialize_database()
        self._initialize_characters()
    
    def _initialize_database(self):
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                character_perspective TEXT,
                memory_type TEXT,
                relationship_type TEXT,
                content TEXT,
                chapter INTEGER,
                timestamp TEXT,
                participants TEXT,
                emotional_weight REAL,
                importance REAL,
                is_secret BOOLEAN,
                contradicts TEXT,
                supports TEXT,
                embedding BLOB
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_edges (
                id TEXT PRIMARY KEY,
                graph_type TEXT,
                source TEXT,
                target TEXT,
                weight REAL,
                metadata TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _initialize_characters(self):
        """Initialize core memory for each character"""
        characters = {
            "Byleth": {
                "personality": "Manipulative, strategic, charming facade",
                "goals": "Maintain multiple secret relationships, avoid detection",
                "secrets": ["Affair with Dimitri", "Affair with Sylvain"],
                "relationships": {"Dimitri": "secret_affair", "Sylvain": "secret_affair", "Annette": "fake_friend"}
            },
            "Dimitri": {
                "personality": "Intense, passionate, possessive when infatuated",
                "goals": "Deepen relationship with Byleth",
                "secrets": ["Affair with Byleth"],
                "relationships": {"Byleth": "secret_affair", "Dedue": "loyal_friend"}
            },
            "Sylvain": {
                "personality": "Charming, flirtatious, easily flattered",
                "goals": "Balance relationship with Annette and attraction to Byleth",
                "secrets": ["Affair with Byleth"],
                "relationships": {"Byleth": "secret_affair", "Annette": "official_relationship"}
            },
            "Annette": {
                "personality": "Trusting, optimistic, becoming suspicious",
                "goals": "Maintain relationship with Sylvain, plan romantic surprises",
                "secrets": [],
                "relationships": {"Sylvain": "official_relationship", "Mercedes": "close_friend"}
            },
            "Dedue": {
                "personality": "Observant, loyal, protective of Dimitri",
                "goals": "Protect Dimitri from scandal and poor decisions",
                "secrets": ["Knows about Byleth-Dimitri affair"],
                "relationships": {"Dimitri": "loyal_protector"}
            }
        }
        
        for char_name, char_data in characters.items():
            self.core_memory[char_name] = char_data
    
    def add_memory(self, 
                   character_perspective: str,
                   memory_type: MemoryType, 
                   relationship_type: RelationshipType,
                   content: str,
                   chapter: int,
                   participants: List[str],
                   emotional_weight: float = 0.0,
                   importance: float = 0.5,
                   is_secret: bool = False) -> str:
        """Add a new memory to the system"""
        
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Generate embedding
        embedding = self.embedder.encode(content)
        
        memory = Memory(
            id=memory_id,
            character_perspective=character_perspective,
            memory_type=memory_type,
            relationship_type=relationship_type,
            content=content,
            chapter=chapter,
            timestamp=timestamp,
            participants=participants,
            emotional_weight=emotional_weight,
            importance=importance,
            is_secret=is_secret,
            contradicts=[],
            supports=[],
            embedding=embedding
        )
        
        # Store memory
        self.memories[memory_id] = memory
        self.character_perspectives[character_perspective].append(memory_id)
        
        # Update graphs
        self._update_graphs(memory)
        
        # Manage hierarchical storage (MemGPT-style)
        self._manage_hierarchical_storage(memory)
        
        # Persist to database
        self._save_memory_to_db(memory)
        
        logger.info(f"Added memory {memory_id} for {character_perspective}")
        return memory_id
    
    def _update_graphs(self, memory: Memory):
        """Update G-Memory style graph structures"""
        
        # Update interaction graph
        for i, participant1 in enumerate(memory.participants):
            for participant2 in memory.participants[i+1:]:
                # Add edge between participants
                edge_weight = memory.importance * (1.0 if not memory.is_secret else 0.5)
                
                if self.interaction_graph.has_edge(participant1, participant2):
                    self.interaction_graph[participant1][participant2]['weight'] += edge_weight
                    self.interaction_graph[participant1][participant2]['interactions'].append(memory.id)
                else:
                    self.interaction_graph.add_edge(
                        participant1, participant2,
                        weight=edge_weight,
                        interactions=[memory.id]
                    )
        
        # Update insight graph for high-importance memories
        if memory.importance > 0.7:
            insight_node = f"insight_{memory.chapter}_{memory.memory_type.value}"
            self.insight_graph.add_node(insight_node, 
                                      memory_ids=[memory.id],
                                      theme=memory.memory_type.value)
    
    def _manage_hierarchical_storage(self, memory: Memory):
        """MemGPT-style memory hierarchy management"""
        
        # Core memory: Essential character information
        if memory.memory_type == MemoryType.CORE or memory.importance > 0.8:
            char_core = self.core_memory.get(memory.character_perspective, {})
            if 'key_memories' not in char_core:
                char_core['key_memories'] = []
            char_core['key_memories'].append(memory.id)
            self.core_memory[memory.character_perspective] = char_core
        
        # Recall memory: Recent interactions (last 20)
        self.recall_memory.append(memory.id)
        if len(self.recall_memory) > 20:
            # Move oldest to archival
            oldest = self.recall_memory.pop(0)
            self.archival_memory.append(oldest)
        
        # Archival memory: Long-term storage with compression
        if len(self.archival_memory) > 100:
            self._compress_archival_memory()
    
    def retrieve_memories(self, 
                         query: str,
                         character_perspective: str = None,
                         relationship_type: RelationshipType = None,
                         top_k: int = 5,
                         chapter_range: Tuple[int, int] = None) -> List[Memory]:
        """Retrieve relevant memories based on query"""
        
        query_embedding = self.embedder.encode(query)
        candidates = []
        
        for memory_id, memory in self.memories.items():
            # Filter by character perspective
            if character_perspective and memory.character_perspective != character_perspective:
                continue
            
            # Filter by relationship type
            if relationship_type and memory.relationship_type != relationship_type:
                continue
            
            # Filter by chapter range
            if chapter_range and not (chapter_range[0] <= memory.chapter <= chapter_range[1]):
                continue
            
            # Calculate similarity
            similarity = cosine_similarity([query_embedding], [memory.embedding])[0][0]
            
            # Weight by importance and recency
            recency_weight = 1.0 / (1.0 + (datetime.now() - memory.timestamp).days)
            final_score = similarity * memory.importance * recency_weight
            
            candidates.append((final_score, memory))
        
        # Sort and return top-k
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in candidates[:top_k]]
    
    def detect_contradictions(self, memory_id: str) -> List[str]:
        """Detect memories that contradict the given memory"""
        memory = self.memories[memory_id]
        contradictions = []
        
        # Check for contradictions within same character's perspective
        for other_id, other_memory in self.memories.items():
            if (other_id != memory_id and 
                other_memory.character_perspective == memory.character_perspective and
                abs(other_memory.chapter - memory.chapter) <= 5):  # Within 5 chapters
                
                # Use embedding similarity to detect potential contradictions
                similarity = cosine_similarity([memory.embedding], [other_memory.embedding])[0][0]
                
                # If very similar content but different emotional weights, might be contradiction
                if (similarity > 0.8 and 
                    abs(memory.emotional_weight - other_memory.emotional_weight) > 1.0):
                    contradictions.append(other_id)
        
        return contradictions
    
    def get_character_relationship_timeline(self, char1: str, char2: str) -> List[Memory]:
        """Get timeline of interactions between two characters"""
        timeline = []
        
        for memory in self.memories.values():
            if char1 in memory.participants and char2 in memory.participants:
                timeline.append(memory)
        
        timeline.sort(key=lambda m: m.chapter)
        return timeline
    
    def analyze_memory_consistency(self) -> Dict[str, Any]:
        """Analyze memory consistency across characters and time"""
        consistency_report = {
            "total_memories": len(self.memories),
            "contradictions": {},
            "character_consistency": {},
            "timeline_gaps": [],
            "secret_exposure_risks": []
        }
        
        # Check for contradictions
        for memory_id in self.memories:
            contradictions = self.detect_contradictions(memory_id)
            if contradictions:
                consistency_report["contradictions"][memory_id] = contradictions
        
        # Character consistency analysis
        for character in self.core_memory.keys():
            char_memories = [m for m in self.memories.values() 
                           if m.character_perspective == character]
            
            # Check for personality drift
            early_memories = [m for m in char_memories if m.chapter <= 10]
            late_memories = [m for m in char_memories if m.chapter >= 40]
            
            if early_memories and late_memories:
                early_sentiment = np.mean([m.emotional_weight for m in early_memories])
                late_sentiment = np.mean([m.emotional_weight for m in late_memories])
                consistency_report["character_consistency"][character] = {
                    "sentiment_drift": abs(early_sentiment - late_sentiment),
                    "memory_count": len(char_memories)
                }
        
        return consistency_report
    
    def _save_memory_to_db(self, memory: Memory):
        """Save memory to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO memories 
            (id, character_perspective, memory_type, relationship_type, content, 
             chapter, timestamp, participants, emotional_weight, importance, 
             is_secret, contradicts, supports, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory.id,
            memory.character_perspective,
            memory.memory_type.value,
            memory.relationship_type.value,
            memory.content,
            memory.chapter,
            memory.timestamp.isoformat(),
            json.dumps(memory.participants),
            memory.emotional_weight,
            memory.importance,
            memory.is_secret,
            json.dumps(memory.contradicts),
            json.dumps(memory.supports),
            memory.embedding.tobytes()
        ))
        
        conn.commit()
        conn.close()
    
    def _compress_archival_memory(self):
        """Compress old memories to save space (MemGPT-style)"""
        # Group memories by theme and timeframe
        # Summarize groups of related memories
        # Keep only essential information
        logger.info("Compressing archival memory...")
        # Implementation would involve LLM-based summarization
    
    def load_chapter_data(self, chapter_data: List[Dict]):
        """Load chapter data and create memories"""
        for chapter_info in chapter_data:
            chapter_num = chapter_info["chapter_number"]
            synopsis = chapter_info["synopsis"]
            
            # Parse synopsis to create memories for different characters
            memories = self._extract_memories_from_synopsis(chapter_num, synopsis)
            
            for memory_data in memories:
                self.add_memory(**memory_data)
    
    def _extract_memories_from_synopsis(self, chapter: int, synopsis: str) -> List[Dict]:
        """Extract character-specific memories from chapter synopsis"""
        memories = []
        
        # This is a simplified version - in practice, you'd want more sophisticated parsing
        # or use an LLM to extract character perspectives
        
        # Identify main characters in synopsis
        characters_mentioned = []
        for char in ["Byleth", "Dimitri", "Sylvain", "Annette", "Dedue", "Mercedes", "Felix"]:
            if char in synopsis:
                characters_mentioned.append(char)
        
        # Create memories based on content patterns
        if "secret" in synopsis.lower() or "affair" in synopsis.lower():
            for char in characters_mentioned:
                memories.append({
                    "character_perspective": char,
                    "memory_type": MemoryType.EPISODIC,
                    "relationship_type": RelationshipType.INTER_CHARACTER,
                    "content": synopsis,
                    "chapter": chapter,
                    "participants": characters_mentioned,
                    "emotional_weight": 0.5 if char == "Byleth" else -0.3,
                    "importance": 0.8,
                    "is_secret": True
                })
        else:
            # Regular memory
            memories.append({
                "character_perspective": characters_mentioned[0] if characters_mentioned else "World",
                "memory_type": MemoryType.EPISODIC,
                "relationship_type": RelationshipType.WORLD_MEMORY,
                "content": synopsis,
                "chapter": chapter,
                "participants": characters_mentioned,
                "emotional_weight": 0.0,
                "importance": 0.5,
                "is_secret": False
            })
        
        return memories


# Evaluation Pipeline
class MemoryEvaluator:
    """Automated evaluation pipeline for the memory system"""
    
    def __init__(self, memory_system: SekaiMemorySystem):
        self.memory_system = memory_system
    
    def evaluate_retrieval_precision_recall(self, test_queries: List[Dict]) -> Dict[str, float]:
        """Evaluate precision and recall of memory retrieval"""
        total_precision = 0.0
        total_recall = 0.0
        
        for query_data in test_queries:
            query = query_data["query"]
            expected_memory_ids = set(query_data["expected_results"])
            
            retrieved_memories = self.memory_system.retrieve_memories(query, top_k=10)
            retrieved_ids = set([m.id for m in retrieved_memories])
            
            if retrieved_ids:
                precision = len(retrieved_ids & expected_memory_ids) / len(retrieved_ids)
                total_precision += precision
            
            if expected_memory_ids:
                recall = len(retrieved_ids & expected_memory_ids) / len(expected_memory_ids)
                total_recall += recall
        
        return {
            "precision": total_precision / len(test_queries),
            "recall": total_recall / len(test_queries)
        }
    
    def evaluate_consistency(self) -> Dict[str, Any]:
        """Evaluate memory consistency across characters and time"""
        return self.memory_system.analyze_memory_consistency()
    
    def evaluate_character_coherence(self) -> Dict[str, float]:
        """Evaluate how well characters maintain consistent personalities"""
        coherence_scores = {}
        
        for character in self.memory_system.core_memory.keys():
            char_memories = [m for m in self.memory_system.memories.values() 
                           if m.character_perspective == character]
            
            if len(char_memories) < 2:
                coherence_scores[character] = 1.0
                continue
            
            # Calculate consistency of emotional weights over time
            weights = [m.emotional_weight for m in sorted(char_memories, key=lambda x: x.chapter)]
            
            # Calculate variance - lower variance = higher coherence
            variance = np.var(weights)
            coherence_score = max(0.0, 1.0 - variance)
            coherence_scores[character] = coherence_score
        
        return coherence_scores


# Example usage and testing
if __name__ == "__main__":
    # Initialize the memory system
    memory_system = SekaiMemorySystem()
    
    # Example chapter data (from your JSON)
    sample_chapter_data = [
        {
            "chapter_number": 1,
            "synopsis": "Byleth Eisner steps into Garreg Mach Corp, the air buzzing with first-day energy. Beneath a professional smile, a sharp mind is already at work, evaluating the office ecosystem. Key players like Dimitri, Sylvain, and Felix are noted - each a potential asset or obstacle in the games to come."
        },
        {
            "chapter_number": 7,
            "synopsis": "Inside the privacy of Dimitri's office, the simmering tension boils over. Byleth makes the first move, initiating a kiss Dimitri returns with unexpected fervor. The professional boundary shatters, replaced by sudden, consuming passion. The affair ignites in the sterile quiet of the corporate building after dark."
        }
    ]
    
    # Load chapter data
    memory_system.load_chapter_data(sample_chapter_data)
    
    # Test memory retrieval
    memories = memory_system.retrieve_memories("Byleth and Dimitri relationship", character_perspective="Byleth")
    print(f"Retrieved {len(memories)} memories about Byleth-Dimitri relationship")
    
    # Analyze consistency
    consistency_report = memory_system.analyze_memory_consistency()
    print(f"Memory consistency report: {consistency_report}")
    
    # Initialize evaluator
    evaluator = MemoryEvaluator(memory_system)
    
    # Character coherence evaluation
    coherence_scores = evaluator.evaluate_character_coherence()
    print(f"Character coherence scores: {coherence_scores}")