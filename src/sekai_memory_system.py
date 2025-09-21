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
import logging
import sqlite3
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os

from .evaluator import MemoryEvaluator
from .models import Memory, MemoryType, RelationshipType
from .storage import (
    initialize_database,
    load_all_character_profiles,
    save_character_profile,
    save_memory_to_db,
)


class SekaiMemorySystem:
    """
    Main memory system implementing MIRIX + G-Memory + MemGPT architecture
    """

    def __init__(self, db_path: str = "sekai_memory.db"):
        self.db_path = db_path
        # Lazy-load sentence transformer to avoid heavy imports on module import
        try:
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            logger.warning(
                "SentenceTransformer not available; embeddings will be zero-vectors."
            )
            self.embedder = None

        # G-Memory inspired graph structures
        self.insight_graph = nx.DiGraph()  # High-level patterns and insights
        self.query_graph = nx.DiGraph()  # Query relationships and dependencies
        self.interaction_graph = nx.DiGraph()  # Direct character interactions

        # Memory storage
        self.memories: Dict[str, Memory] = {}
        self.character_perspectives: Dict[str, List[str]] = defaultdict(list)

        # MemGPT-style hierarchical storage
        self.core_memory: Dict[str, Dict] = {}  # Always accessible
        self.archival_memory: List[str] = []  # Long-term storage
        self.recall_memory: List[str] = []  # Recent interactions

        self._initialize_database()
        # Load initial character profiles from centralized JSON and DB
        self._initialize_characters()
        # Merge any persisted profiles from DB
        try:
            persisted = load_all_character_profiles(self.db_path)
            for name, prof in persisted.items():
                if prof and name not in self.core_memory:
                    self.core_memory[name] = prof
        except Exception:
            logger.debug("No persisted character profiles found or failed to load.")

    def _initialize_database(self):
        """Initialize SQLite database for persistent storage (delegated)."""
        initialize_database(self.db_path)

    def _initialize_characters(self):
        """Initialize core memory for each character"""
        characters = {
            "Byleth": {
                "personality": "Manipulative, strategic, charming facade",
                "goals": "Maintain multiple secret relationships, avoid detection",
                "secrets": ["Affair with Dimitri", "Affair with Sylvain"],
                "relationships": {
                    "Dimitri": "secret_affair",
                    "Sylvain": "secret_affair",
                    "Annette": "fake_friend",
                },
            },
            "Dimitri": {
                "personality": "Intense, passionate, possessive when infatuated",
                "goals": "Deepen relationship with Byleth",
                "secrets": ["Affair with Byleth"],
                "relationships": {"Byleth": "secret_affair", "Dedue": "loyal_friend"},
            },
            "Sylvain": {
                "personality": "Charming, flirtatious, easily flattered",
                "goals": "Balance relationship with Annette and attraction to Byleth",
                "secrets": ["Affair with Byleth"],
                "relationships": {
                    "Byleth": "secret_affair",
                    "Annette": "official_relationship",
                },
            },
            "Annette": {
                "personality": "Trusting, optimistic, becoming suspicious",
                "goals": "Maintain relationship with Sylvain, plan romantic surprises",
                "secrets": [],
                "relationships": {
                    "Sylvain": "official_relationship",
                    "Mercedes": "close_friend",
                },
            },
            "Dedue": {
                "personality": "Observant, loyal, protective of Dimitri",
                "goals": "Protect Dimitri from scandal and poor decisions",
                "secrets": ["Knows about Byleth-Dimitri affair"],
                "relationships": {"Dimitri": "loyal_protector"},
            },
        }

        for char_name, char_data in characters.items():
            self.core_memory[char_name] = char_data

    def ensure_character_registered(
        self, name: str, profile: Optional[Dict[str, Any]] = None
    ):
        """Ensure core structures exist for a character (auto-register new characters)."""
        if not name:
            return
        # Ensure perspectives mapping exists
        if name not in self.character_perspectives:
            self.character_perspectives[name] = []

        # Create a minimal core profile if missing
        if name not in self.core_memory:
            default_profile = profile or {
                "personality": "Unknown",
                "goals": "",
                "secrets": [],
                "relationships": {},
            }
            logger.info(
                f"Auto-registering new character '{name}' with default profile."
            )
            self.core_memory[name] = default_profile
            # Persist the new profile so it survives restarts
            try:
                save_character_profile(self.db_path, name, default_profile)
            except Exception:
                logger.debug("Failed to persist new character profile to DB.")
        return

    def extract_character_names(self, text: str) -> List[str]:
        """Extract character names from text; stub for LLM-based extractor with regex fallback.

        Replace the body with an LLM call (or spaCy NER) later. For now, use a safer regex.
        """
        # Preferred: call out to an LLM or NER pipeline here (not implemented to avoid network calls)
        # Fallback: conservative regex that matches capitalized tokens of reasonable length
        import re

        tokens = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
        # Filter out common stopwords
        stop = {"The", "It", "He", "She", "They", "Chapter"}
        return [t for t in tokens if t not in stop]

    def add_memory(
        self,
        character_perspective: str,
        memory_type: MemoryType,
        relationship_type: RelationshipType,
        content: str,
        chapter: int,
        participants: List[str],
        emotional_weight: float = 0.0,
        importance: float = 0.5,
        is_secret: bool = False,
    ) -> str:
        """Add a new memory to the system"""

        # Ensure perspective and participants are registered so system can grow dynamically
        self.ensure_character_registered(character_perspective)
        for p in participants:
            self.ensure_character_registered(p)

        memory_id = str(uuid.uuid4())
        timestamp = datetime.now()

        # Generate embedding (fallback to zero-vector if embedder missing)
        if self.embedder is not None:
            embedding = self.embedder.encode(content)
        else:
            # all-MiniLM-L6-v2 uses 384-dim embeddings
            embedding = np.zeros(384, dtype=float)

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
            embedding=embedding,
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
            for participant2 in memory.participants[i + 1 :]:
                # Add edge between participants
                edge_weight = memory.importance * (1.0 if not memory.is_secret else 0.5)

                if self.interaction_graph.has_edge(participant1, participant2):
                    self.interaction_graph[participant1][participant2][
                        "weight"
                    ] += edge_weight
                    self.interaction_graph[participant1][participant2][
                        "interactions"
                    ].append(memory.id)
                else:
                    self.interaction_graph.add_edge(
                        participant1,
                        participant2,
                        weight=edge_weight,
                        interactions=[memory.id],
                    )

        # Update insight graph for high-importance memories
        if memory.importance > 0.7:
            insight_node = f"insight_{memory.chapter}_{memory.memory_type.value}"
            self.insight_graph.add_node(
                insight_node, memory_ids=[memory.id], theme=memory.memory_type.value
            )

    def _manage_hierarchical_storage(self, memory: Memory):
        """MemGPT-style memory hierarchy management"""

        # Core memory: Essential character information
        if memory.memory_type == MemoryType.CORE or memory.importance > 0.8:
            char_core = self.core_memory.get(memory.character_perspective, {})
            if "key_memories" not in char_core:
                char_core["key_memories"] = []
            char_core["key_memories"].append(memory.id)
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

    def retrieve_memories(
        self,
        query: str,
        character_perspective: str = None,
        relationship_type: RelationshipType = None,
        top_k: int = 5,
        chapter_range: Tuple[int, int] = None,
    ) -> List[Memory]:
        """Retrieve relevant memories based on query"""

        # Query embedding (fallback to zero-vector if embedder missing)
        if self.embedder is not None:
            query_embedding = self.embedder.encode(query)
        else:
            query_embedding = np.zeros(384, dtype=float)
        candidates = []

        for memory_id, memory in self.memories.items():
            # Filter by character perspective
            if (
                character_perspective
                and memory.character_perspective != character_perspective
            ):
                continue

            # Filter by relationship type
            if relationship_type and memory.relationship_type != relationship_type:
                continue

            # Filter by chapter range
            if chapter_range and not (
                chapter_range[0] <= memory.chapter <= chapter_range[1]
            ):
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
            if (
                other_id != memory_id
                and other_memory.character_perspective == memory.character_perspective
                and abs(other_memory.chapter - memory.chapter) <= 5
            ):  # Within 5 chapters

                # Use embedding similarity to detect potential contradictions
                similarity = cosine_similarity(
                    [memory.embedding], [other_memory.embedding]
                )[0][0]

                # If very similar content but different emotional weights, might be contradiction
                if (
                    similarity > 0.8
                    and abs(memory.emotional_weight - other_memory.emotional_weight)
                    > 1.0
                ):
                    contradictions.append(other_id)

        return contradictions

    def get_character_relationship_timeline(
        self, char1: str, char2: str
    ) -> List[Memory]:
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
            "secret_exposure_risks": [],
        }

        # Check for contradictions
        for memory_id in self.memories:
            contradictions = self.detect_contradictions(memory_id)
            if contradictions:
                consistency_report["contradictions"][memory_id] = contradictions

        # Character consistency analysis
        for character in self.core_memory.keys():
            char_memories = [
                m
                for m in self.memories.values()
                if m.character_perspective == character
            ]

            # Check for personality drift
            early_memories = [m for m in char_memories if m.chapter <= 10]
            late_memories = [m for m in char_memories if m.chapter >= 40]

            if early_memories and late_memories:
                early_sentiment = np.mean([m.emotional_weight for m in early_memories])
                late_sentiment = np.mean([m.emotional_weight for m in late_memories])
                consistency_report["character_consistency"][character] = {
                    "sentiment_drift": abs(early_sentiment - late_sentiment),
                    "memory_count": len(char_memories),
                }

        return consistency_report

    def _save_memory_to_db(self, memory: Memory):
        """Save memory to SQLite database (delegated)."""
        save_memory_to_db(self.db_path, memory)

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

    def _extract_memories_from_synopsis(
        self, chapter: int, synopsis: str
    ) -> List[Dict]:
        """Extract character-specific memories from chapter synopsis"""
        memories = []

        # This is a simplified version - in practice, you'd want more sophisticated parsing
        # or use an LLM to extract character perspectives

        # Identify main characters using central list + LLM/regex extractor
        characters_mentioned = []
        # First, check against existing known characters
        for known in list(self.core_memory.keys()):
            if known in synopsis:
                characters_mentioned.append(known)

        # Then use extractor (LLM stub) to find additional candidate names
        candidates = self.extract_character_names(synopsis)
        for cand in candidates:
            if cand not in characters_mentioned:
                characters_mentioned.append(cand)
                self.ensure_character_registered(cand)

        # Create memories based on content patterns
        if "secret" in synopsis.lower() or "affair" in synopsis.lower():
            for char in characters_mentioned:
                memories.append(
                    {
                        "character_perspective": char,
                        "memory_type": MemoryType.EPISODIC,
                        "relationship_type": RelationshipType.INTER_CHARACTER,
                        "content": synopsis,
                        "chapter": chapter,
                        "participants": characters_mentioned,
                        "emotional_weight": 0.5 if char == "Byleth" else -0.3,
                        "importance": 0.8,
                        "is_secret": True,
                    }
                )
        else:
            # Regular memory
            memories.append(
                {
                    "character_perspective": (
                        characters_mentioned[0] if characters_mentioned else "World"
                    ),
                    "memory_type": MemoryType.EPISODIC,
                    "relationship_type": RelationshipType.WORLD_MEMORY,
                    "content": synopsis,
                    "chapter": chapter,
                    "participants": characters_mentioned,
                    "emotional_weight": 0.0,
                    "importance": 0.5,
                    "is_secret": False,
                }
            )

        return memories


# Evaluation Pipeline is provided by src.evaluator.MemoryEvaluator


# Example usage and testing
if __name__ == "__main__":
    # Initialize the memory system
    memory_system = SekaiMemorySystem()

    # Example chapter data (from your JSON)
    sample_chapter_data = [
        {
            "chapter_number": 1,
            "synopsis": "Byleth Eisner steps into Garreg Mach Corp, the air buzzing with first-day energy. Beneath a professional smile, a sharp mind is already at work, evaluating the office ecosystem. Key players like Dimitri, Sylvain, and Felix are noted - each a potential asset or obstacle in the games to come.",
        },
        {
            "chapter_number": 7,
            "synopsis": "Inside the privacy of Dimitri's office, the simmering tension boils over. Byleth makes the first move, initiating a kiss Dimitri returns with unexpected fervor. The professional boundary shatters, replaced by sudden, consuming passion. The affair ignites in the sterile quiet of the corporate building after dark.",
        },
    ]

    # Load chapter data
    memory_system.load_chapter_data(sample_chapter_data)

    # Test memory retrieval
    memories = memory_system.retrieve_memories(
        "Byleth and Dimitri relationship", character_perspective="Byleth"
    )
    print(f"Retrieved {len(memories)} memories about Byleth-Dimitri relationship")

    # Analyze consistency
    consistency_report = memory_system.analyze_memory_consistency()
    print(f"Memory consistency report: {consistency_report}")

    # Initialize evaluator
    evaluator = MemoryEvaluator(memory_system)

    # Character coherence evaluation
    coherence_scores = evaluator.evaluate_character_coherence()
    print(f"Character coherence scores: {coherence_scores}")
