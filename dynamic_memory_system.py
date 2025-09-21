# dynamic_memory_system.py
"""
Dynamic Memory Growth System
Enables real-time narrative expansion with automatic memory updates,
new character integration, and relationship evolution tracking.

ENHANCED WITH LLM INTEGRATION for sophisticated character analysis
"""

import json
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import networkx as nx
from collections import defaultdict, Counter
import logging
from pathlib import Path
import threading
import time
import hashlib
import asyncio
import os
import re

# OpenAI integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not installed. LLM features will be disabled.")

from sekai_memory_system import SekaiMemorySystem, Memory, MemoryType, RelationshipType, MemoryPerspective
from data_loader import SekaiDataLoader, MemoryDataProcessor
from env_config import env_config

logger = logging.getLogger(__name__)

@dataclass
class MemoryUpdate:
    """Represents a memory update operation"""
    update_id: str
    update_type: str  # 'new_memory', 'character_added', 'relationship_changed', 'contradiction_resolved'
    timestamp: datetime
    affected_memories: List[str]
    new_memories: List[str]
    metadata: Dict[str, Any]

@dataclass
class CharacterProfile:
    """Dynamic character profile that evolves"""
    name: str
    introduction_chapter: int
    personality_traits: List[str]
    relationship_history: Dict[str, List[Tuple[int, str, float]]]  # character -> [(chapter, type, strength)]
    secret_knowledge: Set[str]  # memory IDs of secrets they know
    memory_patterns: Dict[str, float]  # behavioral patterns
    last_updated: datetime
    llm_analysis_confidence: float = 0.0  # NEW: LLM analysis confidence
    behavioral_patterns: List[str] = None  # NEW: LLM-derived patterns
    emotional_state: str = "unknown"  # NEW: Current emotional state
    character_arc_stage: str = "developing"  # NEW: Development stage

class LLMConfig:
    """Configuration for OpenAI LLM integration using environment variables"""
    def __init__(self, api_key: str = None, model: str = None, temperature: float = None):
        # Use provided values or fall back to environment config
        self.api_key = api_key or env_config.openai_api_key
        self.model = model or env_config.openai_model
        self.temperature = temperature if temperature is not None else env_config.openai_temperature
        self.max_tokens = env_config.openai_max_tokens
        
        # Updated for OpenAI v1.0+ - no need to set global api_key
        if self.api_key and OPENAI_AVAILABLE:
            self.enabled = True
            logger.info(f"LLM features enabled with model: {self.model}")
        else:
            self.enabled = False
            if not self.api_key:
                logger.warning("LLM features disabled: No OpenAI API key found in environment variables or .env file")
            else:
                logger.warning("LLM features disabled: OpenAI library not available")

class DynamicMemorySystem(SekaiMemorySystem):
    """Enhanced memory system with dynamic growth capabilities and LLM integration"""
    
    def __init__(self, db_path: str = None, watch_directory: str = None, 
                 openai_api_key: str = None, llm_model: str = None):
        # Use environment config as defaults
        db_path = db_path or env_config.db_path
        watch_directory = watch_directory or env_config.watch_directory
        
        super().__init__(db_path)
        
        # Dynamic growth components
        self.character_profiles: Dict[str, CharacterProfile] = {}
        self.memory_updates: List[MemoryUpdate] = []
        self.relationship_graph = nx.Graph()  # Dynamic relationship tracking
        self.narrative_timeline: Dict[int, List[str]] = defaultdict(list)  # chapter -> memory_ids
        
        # File watching for automatic updates
        self.watch_directory = Path(watch_directory) if watch_directory else None
        self.watched_files: Dict[str, float] = {}  # file -> last_modified_time
        self.file_watcher_active = False
        
        # Memory consolidation settings from environment
        self.consolidation_threshold = env_config.consolidation_threshold
        
        # LLM Integration with environment configuration
        self.llm_config = LLMConfig(openai_api_key, llm_model)
        
        # Initialize components (now with LLM enhancement)
        self.contradiction_resolver = ContradictionResolver(self)
        self.relationship_tracker = RelationshipTracker(self)
        self.new_character_integrator = NewCharacterIntegrator(self)
        
        # Initialize dynamic profiles from existing core memory
        self._initialize_dynamic_profiles()
        
        logger.info(f"Dynamic Memory System initialized with {len(self.character_profiles)} character profiles")
        if self.llm_config.enabled:
            logger.info(f"LLM integration enabled with model: {self.llm_config.model}")
        else:
            logger.info("LLM integration disabled - using statistical analysis fallbacks")
    
    def _initialize_dynamic_profiles(self):
        """Initialize dynamic character profiles from existing data"""
        for character_name, core_data in self.core_memory.items():
            profile = CharacterProfile(
                name=character_name,
                introduction_chapter=1,  # Default, will be updated
                personality_traits=core_data.get('personality', '').split(', ') if core_data.get('personality') else [],
                relationship_history=defaultdict(list),
                secret_knowledge=set(),
                memory_patterns={},
                last_updated=datetime.now(),
                behavioral_patterns=[]
            )
            
            # Analyze existing memories to build profile
            char_memories = [m for m in self.memories.values() if m.character_perspective == character_name]
            if char_memories:
                profile.introduction_chapter = min(m.chapter for m in char_memories)
                
                # Build relationship history
                for memory in char_memories:
                    for participant in memory.participants:
                        if participant != character_name:
                            profile.relationship_history[participant].append(
                                (memory.chapter, memory.relationship_type.value, memory.emotional_weight)
                            )
                    
                    # Track secrets
                    if memory.is_secret:
                        profile.secret_knowledge.add(memory.id)
            
            self.character_profiles[character_name] = profile
    
    async def _llm_analyze_character(self, character_name: str, memories: List[Memory]) -> Dict[str, Any]:
        """Use LLM to analyze character traits and development"""
        if not self.llm_config.enabled or not memories:
            return self._fallback_character_analysis(character_name, memories)
        
        # Prepare memory context for LLM
        memory_contexts = []
        for memory in memories[-10:]:  # Use recent memories
            memory_contexts.append({
                "chapter": memory.chapter,
                "content": memory.content,
                "emotional_weight": memory.emotional_weight,
                "participants": memory.participants,
                "is_secret": memory.is_secret
            })
        
        context_text = "\n".join([
            f"Chapter {ctx['chapter']}: {ctx['content']} "
            f"(Emotional weight: {ctx['emotional_weight']}, Secret: {ctx['is_secret']})"
            for ctx in memory_contexts
        ])
        
        prompt = f"""
Analyze the character {character_name} based on these narrative memories:

{context_text}

Provide analysis in JSON format:
{{
    "personality_traits": ["trait1", "trait2", "trait3"],
    "behavioral_patterns": ["pattern1", "pattern2"],
    "emotional_state": "current emotional state",
    "character_arc_stage": "development stage",
    "confidence_score": 0.85
}}

Focus on:
1. Core personality traits that remain consistent
2. Behavioral patterns across interactions
3. Current emotional trajectory
4. Development stage in their character arc
5. Analysis confidence (0.0-1.0)
"""
        
        try:
            # Updated for OpenAI v1.0+ API
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.llm_config.api_key)
            
            response = await client.chat.completions.create(
                model=self.llm_config.model,
                messages=[
                    {"role": "system", "content": "You are an expert narrative psychologist analyzing fictional characters."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.llm_config.max_tokens,
                temperature=self.llm_config.temperature
            )
            
            analysis_text = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
        except Exception as e:
            logger.error(f"LLM character analysis failed for {character_name}: {e}")
        
        return self._fallback_character_analysis(character_name, memories)
    
    def _fallback_character_analysis(self, character_name: str, memories: List[Memory]) -> Dict[str, Any]:
        """Fallback character analysis when LLM is unavailable"""
        if not memories:
            return {
                "personality_traits": ["developing"],
                "behavioral_patterns": ["unknown"],
                "emotional_state": "unknown",
                "character_arc_stage": "introduction",
                "confidence_score": 0.3
            }
        
        # Simple statistical analysis
        emotional_weights = [m.emotional_weight for m in memories]
        avg_emotion = np.mean(emotional_weights)
        emotion_variance = np.var(emotional_weights)
        
        traits = []
        if avg_emotion > 0.3:
            traits.append("optimistic")
        elif avg_emotion < -0.3:
            traits.append("pessimistic")
        
        if emotion_variance > 0.8:
            traits.append("emotionally_volatile")
        else:
            traits.append("emotionally_stable")
        
        secret_ratio = len([m for m in memories if m.is_secret]) / len(memories)
        if secret_ratio > 0.5:
            traits.append("secretive")
        
        return {
            "personality_traits": traits,
            "behavioral_patterns": ["pattern_based_on_interactions"],
            "emotional_state": "positive" if avg_emotion > 0 else "negative" if avg_emotion < 0 else "neutral",
            "character_arc_stage": "developing",
            "confidence_score": 0.6
        }
    
    async def _llm_consolidate_memories(self, memory_group: List[Memory]) -> Dict[str, Any]:
        """Use LLM to intelligently consolidate memories"""
        if not self.llm_config.enabled or len(memory_group) < 2:
            return self._basic_memory_consolidation(memory_group)
        
        memory_texts = [m.content for m in memory_group]
        participants = list(set([p for m in memory_group for p in m.participants]))
        chapter_range = [min(m.chapter for m in memory_group), max(m.chapter for m in memory_group)]
        
        prompt = f"""
Consolidate these related memories into a coherent summary:

Memories:
{chr(10).join([f"{i+1}. {text}" for i, text in enumerate(memory_texts)])}

Context: Chapters {chapter_range[0]}-{chapter_range[1]}, Characters: {', '.join(participants)}

Return JSON:
{{
    "consolidated_content": "comprehensive summary preserving key details",
    "key_themes": ["theme1", "theme2"],
    "emotional_progression": "how emotions evolved",
    "critical_details": ["important detail1", "important detail2"]
}}
"""
        
        try:
            # Updated for OpenAI v1.0+ API
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.llm_config.api_key)
            
            response = await client.chat.completions.create(
                model=self.llm_config.model,
                messages=[
                    {"role": "system", "content": "You are a narrative consolidation specialist."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=self.llm_config.temperature
            )
            
            result_text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group())
                
        except Exception as e:
            logger.error(f"LLM memory consolidation failed: {e}")
        
        return self._basic_memory_consolidation(memory_group)
    
    def _basic_memory_consolidation(self, memory_group: List[Memory]) -> Dict[str, Any]:
        """Basic memory consolidation without LLM"""
        participants = list(set([p for m in memory_group for p in m.participants]))
        return {
            "consolidated_content": f"Summary of {len(memory_group)} interactions involving {', '.join(participants)}",
            "key_themes": ["interaction", "relationship"],
            "emotional_progression": "varied",
            "critical_details": ["multiple encounters"]
        }
    
    async def _llm_resolve_contradiction(self, memory1: Memory, memory2: Memory) -> Dict[str, Any]:
        """Use LLM to analyze and resolve memory contradictions"""
        if not self.llm_config.enabled:
            return self._basic_contradiction_resolution(memory1, memory2)
        
        prompt = f"""
Analyze these potentially contradictory memories:

Memory 1 (Chapter {memory1.chapter}): "{memory1.content}"
- Character: {memory1.character_perspective}, Emotion: {memory1.emotional_weight}

Memory 2 (Chapter {memory2.chapter}): "{memory2.content}"
- Character: {memory2.character_perspective}, Emotion: {memory2.emotional_weight}

Return JSON:
{{
    "is_contradiction": true/false,
    "contradiction_type": "factual/emotional/perspective/temporal",
    "resolution_strategy": "character_development/different_perspectives/factual_error/time_progression",
    "explanation": "detailed explanation",
    "confidence": 0.85,
    "action": "keep_both/prefer_newer/prefer_higher_importance/merge"
}}
"""
        
        try:
            # Updated for OpenAI v1.0+ API
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.llm_config.api_key)
            
            response = await client.chat.completions.create(
                model=self.llm_config.model,
                messages=[
                    {"role": "system", "content": "You are a narrative consistency analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.2  # Lower temperature for consistency analysis
            )
            
            result_text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group())
                
        except Exception as e:
            logger.error(f"LLM contradiction resolution failed: {e}")
        
        return self._basic_contradiction_resolution(memory1, memory2)
    
    def _basic_contradiction_resolution(self, memory1: Memory, memory2: Memory) -> Dict[str, Any]:
        """Basic contradiction resolution without LLM"""
        if memory2.chapter > memory1.chapter + 5:
            return {
                "is_contradiction": True,
                "contradiction_type": "temporal",
                "resolution_strategy": "character_development",
                "explanation": "Character evolved over time",
                "confidence": 0.7,
                "action": "keep_both"
            }
        elif memory2.importance > memory1.importance * 1.2:
            return {
                "is_contradiction": True,
                "contradiction_type": "importance",
                "resolution_strategy": "prefer_important",
                "explanation": "Higher importance memory preferred",
                "confidence": 0.8,
                "action": "prefer_newer"
            }
        else:
            return {
                "is_contradiction": False,
                "contradiction_type": "none",
                "resolution_strategy": "no_action",
                "explanation": "No significant contradiction",
                "confidence": 0.6,
                "action": "keep_both"
            }
    
    async def add_new_chapter(self, chapter_data: Dict[str, Any], auto_integrate: bool = True) -> Dict[str, Any]:
        """Add a new chapter with full dynamic integration and LLM enhancement"""
        logger.info(f"Adding new chapter {chapter_data.get('chapter_number', 'unknown')}")
        
        start_time = datetime.now()
        integration_results = {
            'chapter_number': chapter_data.get('chapter_number'),
            'new_memories_created': 0,
            'new_characters_discovered': [],
            'relationship_changes': [],
            'contradictions_detected': [],
            'contradictions_resolved': [],
            'processing_time': 0.0,
            'update_id': str(uuid.uuid4()),
            'llm_enhanced': self.llm_config.enabled
        }
        
        # Extract memories from new chapter
        data_loader = SekaiDataLoader()
        extracted_memories = data_loader.extract_memories_from_chapter(chapter_data)
        
        # Identify new characters
        mentioned_characters = set()
        for memory_data in extracted_memories:
            mentioned_characters.update(memory_data.participants)
        
        new_characters = mentioned_characters - set(self.character_profiles.keys())
        if new_characters:
            logger.info(f"Discovered new characters: {list(new_characters)}")
            integration_results['new_characters_discovered'] = list(new_characters)
            
            if auto_integrate:
                await self._integrate_new_characters(new_characters, chapter_data['chapter_number'])
        
        # Add memories to system
        new_memory_ids = []
        for memory_data in extracted_memories:
            memory_id = self.add_memory(
                character_perspective=memory_data.character_perspective,
                memory_type=memory_data.memory_type,
                relationship_type=memory_data.relationship_type,
                content=memory_data.content,
                chapter=chapter_data['chapter_number'],
                participants=memory_data.participants,
                emotional_weight=memory_data.emotional_weight,
                importance=memory_data.importance,
                is_secret=memory_data.is_secret
            )
            new_memory_ids.append(memory_id)
        
        integration_results['new_memories_created'] = len(new_memory_ids)
        
        # Update narrative timeline
        chapter_num = chapter_data['chapter_number']
        self.narrative_timeline[chapter_num].extend(new_memory_ids)
        
        if auto_integrate:
            # Enhanced contradiction detection and resolution
            contradictions = self._detect_new_contradictions(new_memory_ids)
            integration_results['contradictions_detected'] = contradictions
            
            if contradictions:
                resolutions = await self._resolve_contradictions_enhanced(contradictions)
                integration_results['contradictions_resolved'] = resolutions
            
            # Update relationship tracking
            relationship_changes = self.relationship_tracker.update_relationships(new_memory_ids)
            integration_results['relationship_changes'] = relationship_changes
            
            # Enhanced character profile updates
            await self._update_character_profiles_enhanced(new_memory_ids, mentioned_characters)
            
            # Trigger memory consolidation if needed
            if len(new_memory_ids) >= self.consolidation_threshold:
                await self._trigger_memory_consolidation_enhanced(chapter_num)
        
        # Record this update
        update = MemoryUpdate(
            update_id=integration_results['update_id'],
            update_type='new_chapter',
            timestamp=start_time,
            affected_memories=[],
            new_memories=new_memory_ids,
            metadata=integration_results
        )
        self.memory_updates.append(update)
        
        integration_results['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Chapter integration completed in {integration_results['processing_time']:.2f}s")
        return integration_results
    
    async def _integrate_new_characters(self, new_characters: Set[str], introduction_chapter: int):
        """Integrate newly discovered characters with LLM analysis"""
        for character_name in new_characters:
            logger.info(f"Integrating new character: {character_name}")
            
            # Create character profile
            profile = CharacterProfile(
                name=character_name,
                introduction_chapter=introduction_chapter,
                personality_traits=[],  # Will be inferred
                relationship_history=defaultdict(list),
                secret_knowledge=set(),
                memory_patterns={},
                last_updated=datetime.now(),
                behavioral_patterns=[]
            )
            
            # Add to core memory with default settings
            self.core_memory[character_name] = {
                "personality": "Unknown - recently introduced",
                "goals": "To be determined from interactions",
                "secrets": [],
                "relationships": {}
            }
            
            self.character_profiles[character_name] = profile
            
            # Enhanced character trait inference with LLM
            await self._infer_character_traits_enhanced(character_name)
    
    async def _infer_character_traits_enhanced(self, character_name: str):
        """Enhanced character trait inference using LLM"""
        char_memories = [
            m for m in self.memories.values() 
            if m.character_perspective == character_name
        ]
        
        if not char_memories:
            return
        
        # Use LLM for character analysis
        analysis_result = await self._llm_analyze_character(character_name, char_memories)
        
        # Update character profile with LLM insights
        profile = self.character_profiles[character_name]
        profile.personality_traits = analysis_result["personality_traits"]
        profile.behavioral_patterns = analysis_result["behavioral_patterns"]
        profile.emotional_state = analysis_result["emotional_state"]
        profile.character_arc_stage = analysis_result["character_arc_stage"]
        profile.llm_analysis_confidence = analysis_result["confidence_score"]
        
        # Update core memory
        self.core_memory[character_name].update({
            "personality": ", ".join(analysis_result["personality_traits"]),
            "behavioral_patterns": analysis_result["behavioral_patterns"],
            "emotional_state": analysis_result["emotional_state"],
            "llm_confidence": analysis_result["confidence_score"]
        })
        
        logger.info(f"Enhanced trait inference for {character_name}: {analysis_result['personality_traits']} "
                   f"(confidence: {analysis_result['confidence_score']:.2f})")
    
    async def _resolve_contradictions_enhanced(self, contradictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced contradiction resolution using LLM"""
        resolutions = []
        
        for contradiction in contradictions:
            memory1 = self.memories[contradiction['new_memory_id']]
            memory2 = self.memories[contradiction['conflicting_memory_id']]
            
            resolution = await self._llm_resolve_contradiction(memory1, memory2)
            
            # Apply resolution based on LLM analysis
            if resolution["action"] == "prefer_newer":
                memory1.supports.append(memory2.id)
                memory2.contradicts.append(memory1.id)
            elif resolution["action"] == "keep_both":
                memory1.supports.append(memory2.id)
                memory2.supports.append(memory1.id)
            
            resolution['contradiction_metadata'] = contradiction
            resolutions.append(resolution)
        
        return resolutions
    
    async def _update_character_profiles_enhanced(self, new_memory_ids: List[str], mentioned_characters: Set[str]):
        """Enhanced character profile updates with LLM analysis"""
        for character_name in mentioned_characters:
            if character_name not in self.character_profiles:
                continue
            
            # Get character's new memories
            char_new_memories = [
                self.memories[mid] for mid in new_memory_ids 
                if self.memories[mid].character_perspective == character_name
            ]
            
            if not char_new_memories:
                continue
            
            # Update relationship history
            profile = self.character_profiles[character_name]
            for memory in char_new_memories:
                for participant in memory.participants:
                    if participant != character_name:
                        profile.relationship_history[participant].append(
                            (memory.chapter, memory.relationship_type.value, memory.emotional_weight)
                        )
                
                # Track new secrets
                if memory.is_secret:
                    profile.secret_knowledge.add(memory.id)
            
            # Enhanced analysis if significant new information
            if len(char_new_memories) >= 2:  # Multiple new memories warrant re-analysis
                all_char_memories = [m for m in self.memories.values() if m.character_perspective == character_name]
                analysis_result = await self._llm_analyze_character(character_name, all_char_memories)
                
                # Update profile with new insights
                profile.personality_traits = analysis_result["personality_traits"]
                profile.behavioral_patterns = analysis_result["behavioral_patterns"]
                profile.emotional_state = analysis_result["emotional_state"]
                profile.character_arc_stage = analysis_result["character_arc_stage"]
                profile.llm_analysis_confidence = analysis_result["confidence_score"]
            
            profile.last_updated = datetime.now()
    
    async def _trigger_memory_consolidation_enhanced(self, chapter_num: int):
        """Enhanced memory consolidation using LLM"""
        logger.info(f"Triggering enhanced memory consolidation for chapter {chapter_num}")
        
        # Find memories that can be consolidated
        recent_memories = []
        for i in range(max(1, chapter_num - 5), chapter_num + 1):
            recent_memories.extend([
                self.memories[mid] for mid in self.narrative_timeline.get(i, [])
            ])
        
        # Group similar memories for consolidation
        consolidation_groups = self._group_memories_for_consolidation(recent_memories)
        
        for group in consolidation_groups:
            if len(group) >= 3:  # Only consolidate if multiple similar memories
                await self._consolidate_memory_group_enhanced(group)
    
    async def _consolidate_memory_group_enhanced(self, memory_group: List[Memory]):
        """Enhanced memory group consolidation using LLM"""
        logger.info(f"Consolidating {len(memory_group)} similar memories with LLM")
        
        # Use LLM for intelligent consolidation
        consolidation_result = await self._llm_consolidate_memories(memory_group)
        
        # Calculate consolidated properties
        avg_emotional_weight = np.mean([m.emotional_weight for m in memory_group])
        max_importance = max(m.importance for m in memory_group)
        all_participants = set()
        for m in memory_group:
            all_participants.update(m.participants)
        
        # Create consolidated memory
        consolidated_id = self.add_memory(
            character_perspective=memory_group[0].character_perspective,
            memory_type=MemoryType.SEMANTIC,  # Consolidated memories are semantic
            relationship_type=memory_group[0].relationship_type,
            content=consolidation_result["consolidated_content"],
            chapter=max(m.chapter for m in memory_group),
            participants=list(all_participants),
            emotional_weight=avg_emotional_weight,
            importance=max_importance,
            is_secret=any(m.is_secret for m in memory_group)
        )
        
        # Add consolidation metadata
        consolidated_memory = self.memories[consolidated_id]
        consolidated_memory.metadata = {
            "consolidation_type": "llm_enhanced",
            "original_count": len(memory_group),
            "key_themes": consolidation_result["key_themes"],
            "emotional_progression": consolidation_result["emotional_progression"],
            "critical_details": consolidation_result["critical_details"],
            "original_memory_ids": [m.id for m in memory_group]
        }
        
        # Move original memories to archival
        for memory in memory_group:
            if memory.id in self.recall_memory:
                self.recall_memory.remove(memory.id)
            if memory.id not in self.archival_memory:
                self.archival_memory.append(memory.id)

    # All other methods from original dynamic_memory_system.py remain the same
    # (file watcher, relationship tracking, etc.)
    
    def _detect_new_contradictions(self, new_memory_ids: List[str]) -> List[Dict[str, Any]]:
        """Detect contradictions introduced by new memories"""
        contradictions = []
        
        for memory_id in new_memory_ids:
            memory = self.memories[memory_id]
            
            # Check against existing memories from same character
            existing_contradictions = self.detect_contradictions(memory_id)
            
            for contradiction_id in existing_contradictions:
                contradiction_memory = self.memories[contradiction_id]
                contradictions.append({
                    'new_memory_id': memory_id,
                    'conflicting_memory_id': contradiction_id,
                    'character': memory.character_perspective,
                    'conflict_type': 'emotional_inconsistency',
                    'new_chapter': memory.chapter,
                    'old_chapter': contradiction_memory.chapter,
                    'severity': abs(memory.emotional_weight - contradiction_memory.emotional_weight)
                })
        
        return contradictions
    
    def get_character_evolution(self, character_name: str) -> Dict[str, Any]:
        """Get detailed character evolution over time with LLM insights"""
        if character_name not in self.character_profiles:
            return {"error": f"Character {character_name} not found"}
        
        profile = self.character_profiles[character_name]
        char_memories = [
            m for m in self.memories.values() 
            if m.character_perspective == character_name
        ]
        char_memories.sort(key=lambda m: m.chapter)
        
        evolution = {
            "character_name": character_name,
            "introduction_chapter": profile.introduction_chapter,
            "total_memories": len(char_memories),
            "chapter_range": [char_memories[0].chapter, char_memories[-1].chapter] if char_memories else [0, 0],
            "relationship_development": {},
            "emotional_journey": [],
            "secret_knowledge_progression": [],
            "personality_evolution": profile.memory_patterns,
            # Enhanced with LLM data
            "llm_analysis": {
                "personality_traits": profile.personality_traits,
                "behavioral_patterns": profile.behavioral_patterns if profile.behavioral_patterns else [],
                "emotional_state": profile.emotional_state,
                "character_arc_stage": profile.character_arc_stage,
                "analysis_confidence": profile.llm_analysis_confidence
            }
        }
        
        # Track relationship development
        for other_char, interactions in profile.relationship_history.items():
            interactions.sort(key=lambda x: x[0])  # Sort by chapter
            evolution["relationship_development"][other_char] = {
                "first_interaction": interactions[0][0] if interactions else None,
                "interaction_count": len(interactions),
                "emotional_progression": [x[2] for x in interactions],
                "relationship_types": list(set(x[1] for x in interactions))
            }
        
        # Track emotional journey
        for memory in char_memories:
            evolution["emotional_journey"].append({
                "chapter": memory.chapter,
                "emotional_weight": memory.emotional_weight,
                "importance": memory.importance,
                "is_secret": memory.is_secret
            })
        
        # Track secret knowledge progression
        secret_memories = [m for m in char_memories if m.is_secret]
        for memory in secret_memories:
            evolution["secret_knowledge_progression"].append({
                "chapter": memory.chapter,
                "secret_type": "affair" if "affair" in memory.content.lower() else "other",
                "participants": memory.participants
            })
        
        return evolution
    
    def get_narrative_growth_stats(self) -> Dict[str, Any]:
        """Get statistics about how the narrative has grown with LLM insights"""
        all_memories = list(self.memories.values())
        
        if not all_memories:
            return {"error": "No memories in system"}
        
        chapters = sorted(set(m.chapter for m in all_memories))
        
        stats = {
            "total_chapters": len(chapters),
            "chapter_range": [min(chapters), max(chapters)],
            "total_memories": len(all_memories),
            "total_characters": len(self.character_profiles),
            "memory_growth_by_chapter": {},
            "character_introduction_timeline": {},
            "relationship_network_growth": {},
            "secret_accumulation": {},
            "recent_updates": len([u for u in self.memory_updates if u.timestamp > datetime.now() - timedelta(days=7)]),
            "llm_enhancement_stats": {
                "llm_enabled": self.llm_config.enabled,
                "characters_with_llm_analysis": 0,
                "average_analysis_confidence": 0.0,
                "llm_model_used": self.llm_config.model if self.llm_config.enabled else None
            }
        }
        
        # LLM enhancement statistics
        if self.llm_config.enabled:
            llm_analyzed_chars = [
                char for char, profile in self.character_profiles.items() 
                if profile.llm_analysis_confidence > 0
            ]
            stats["llm_enhancement_stats"]["characters_with_llm_analysis"] = len(llm_analyzed_chars)
            
            if llm_analyzed_chars:
                confidences = [
                    self.character_profiles[char].llm_analysis_confidence 
                    for char in llm_analyzed_chars
                ]
                stats["llm_enhancement_stats"]["average_analysis_confidence"] = np.mean(confidences)
        
        # Memory growth by chapter
        for chapter in chapters:
            chapter_memories = [m for m in all_memories if m.chapter == chapter]
            stats["memory_growth_by_chapter"][chapter] = {
                "memory_count": len(chapter_memories),
                "new_characters": len(set(m.character_perspective for m in chapter_memories) - 
                                   set(m.character_perspective for m in all_memories if m.chapter < chapter)),
                "secrets_introduced": len([m for m in chapter_memories if m.is_secret])
            }
        
        # Character introduction timeline with LLM insights
        for char_name, profile in self.character_profiles.items():
            stats["character_introduction_timeline"][char_name] = {
                "introduction_chapter": profile.introduction_chapter,
                "total_memories": len([m for m in all_memories if m.character_perspective == char_name]),
                "relationships_formed": len(profile.relationship_history),
                "llm_traits": profile.personality_traits if profile.personality_traits else [],
                "emotional_state": profile.emotional_state,
                "analysis_confidence": profile.llm_analysis_confidence
            }
        
        # Relationship network growth
        relationship_counts = defaultdict(int)
        for memory in all_memories:
            if len(memory.participants) >= 2:
                for i, char1 in enumerate(memory.participants):
                    for char2 in memory.participants[i+1:]:
                        key = tuple(sorted([char1, char2]))
                        relationship_counts[key] += 1
        
        stats["relationship_network_growth"] = {
            "total_relationships": len(relationship_counts),
            "strongest_relationships": sorted(
                relationship_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        }
        
        # Secret accumulation over time
        secrets_by_chapter = defaultdict(int)
        for memory in all_memories:
            if memory.is_secret:
                secrets_by_chapter[memory.chapter] += 1
        
        stats["secret_accumulation"] = dict(secrets_by_chapter)
        
        return stats

    # Rest of the methods remain the same as original dynamic_memory_system.py
    # File watcher methods, etc.
    
    def start_file_watcher(self, check_interval: int = 30):
        """Start watching for new chapter files"""
        if not self.watch_directory or self.file_watcher_active:
            return
        
        self.file_watcher_active = True
        
        def watch_files():
            logger.info(f"Starting file watcher for {self.watch_directory}")
            
            while self.file_watcher_active:
                try:
                    # Check for new or modified JSON files
                    for json_file in self.watch_directory.glob("*.json"):
                        current_mtime = json_file.stat().st_mtime
                        
                        if (str(json_file) not in self.watched_files or 
                            current_mtime > self.watched_files[str(json_file)]):
                            
                            logger.info(f"Detected new/modified file: {json_file}")
                            self.watched_files[str(json_file)] = current_mtime
                            
                            # Process the file
                            try:
                                with open(json_file, 'r') as f:
                                    data = json.load(f)
                                
                                # Handle both single chapter and multi-chapter files
                                if isinstance(data, list):
                                    # Multiple chapters
                                    for chapter_data in data:
                                        if self._is_new_chapter(chapter_data):
                                            # Use async context for LLM features
                                            asyncio.create_task(self.add_new_chapter(chapter_data))
                                elif isinstance(data, dict) and 'chapter_number' in data:
                                    # Single chapter
                                    if self._is_new_chapter(data):
                                        asyncio.create_task(self.add_new_chapter(data))
                                        
                            except Exception as e:
                                logger.error(f"Error processing file {json_file}: {e}")
                    
                    time.sleep(check_interval)
                    
                except Exception as e:
                    logger.error(f"File watcher error: {e}")
                    time.sleep(check_interval)
        
        # Start watcher in background thread
        watcher_thread = threading.Thread(target=watch_files, daemon=True)
        watcher_thread.start()
        
        logger.info("File watcher started")
    
    def stop_file_watcher(self):
        """Stop the file watcher"""
        self.file_watcher_active = False
        logger.info("File watcher stopped")
    
    def _is_new_chapter(self, chapter_data: Dict[str, Any]) -> bool:
        """Check if this is a genuinely new chapter"""
        chapter_num = chapter_data.get('chapter_number')
        
        if not chapter_num:
            return False
        
        # Check if we already have memories for this chapter
        existing_memories = self.narrative_timeline.get(chapter_num, [])
        
        if existing_memories:
            # Check content hash to see if it's actually different
            content_hash = hashlib.md5(chapter_data.get('synopsis', '').encode()).hexdigest()
            
            # Store hash in metadata for comparison
            for memory_id in existing_memories:
                memory = self.memories.get(memory_id)
                if (memory and 
                    hasattr(memory, 'content_hash') and 
                    memory.content_hash == content_hash):
                    return False
        
        return True
    
    def _group_memories_for_consolidation(self, memories: List[Memory]) -> List[List[Memory]]:
        """Group similar memories that can be consolidated"""
        groups = []
        
        # Group by character and relationship context
        char_groups = defaultdict(list)
        for memory in memories:
            key = (memory.character_perspective, tuple(sorted(memory.participants)))
            char_groups[key].append(memory)
        
        # Find groups with multiple similar memories
        for group_memories in char_groups.values():
            if len(group_memories) >= 2:
                # Check semantic similarity
                embeddings = [m.embedding for m in group_memories]
                
                similar_groups = []
                for i, memory1 in enumerate(group_memories):
                    similar_group = [memory1]
                    for j, memory2 in enumerate(group_memories[i+1:], i+1):
                        similarity = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                        )
                        if similarity > 0.8:  # High similarity threshold
                            similar_group.append(memory2)
                    
                    if len(similar_group) >= 2:
                        similar_groups.append(similar_group)
                
                groups.extend(similar_groups)
        
        return groups


class ContradictionResolver:
    """Enhanced contradiction resolver with LLM capabilities"""
    
    def __init__(self, memory_system: DynamicMemorySystem):
        self.memory_system = memory_system
    
    async def resolve_contradictions(self, contradictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve detected contradictions using enhanced methods"""
        resolutions = []
        
        for contradiction in contradictions:
            if self.memory_system.llm_config.enabled:
                resolution = await self._resolve_with_llm(contradiction)
            else:
                resolution = self._resolve_basic(contradiction)
            
            if resolution:
                resolutions.append(resolution)
        
        return resolutions
    
    async def _resolve_with_llm(self, contradiction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Use LLM to resolve contradiction"""
        new_memory = self.memory_system.memories[contradiction['new_memory_id']]
        old_memory = self.memory_system.memories[contradiction['conflicting_memory_id']]
        
        return await self.memory_system._llm_resolve_contradiction(new_memory, old_memory)
    
    def _resolve_basic(self, contradiction: Dict[str, Any]) -> Dict[str, Any]:
        """Basic contradiction resolution"""
        new_memory = self.memory_system.memories[contradiction['new_memory_id']]
        old_memory = self.memory_system.memories[contradiction['conflicting_memory_id']]
        
        return self.memory_system._basic_contradiction_resolution(new_memory, old_memory)


class RelationshipTracker:
    """Enhanced relationship tracker (methods remain same as original)"""
    
    def __init__(self, memory_system: DynamicMemorySystem):
        self.memory_system = memory_system
    
    def update_relationships(self, new_memory_ids: List[str]) -> List[Dict[str, Any]]:
        """Update relationship tracking based on new memories"""
        relationship_changes = []
        
        for memory_id in new_memory_ids:
            memory = self.memory_system.memories[memory_id]
            
            if len(memory.participants) >= 2:
                changes = self._analyze_relationship_changes(memory)
                relationship_changes.extend(changes)
        
        return relationship_changes
    
    def _analyze_relationship_changes(self, memory: Memory) -> List[Dict[str, Any]]:
        """Analyze relationship changes from a single memory"""
        changes = []
        
        for i, char1 in enumerate(memory.participants):
            for char2 in memory.participants[i+1:]:
                # Get relationship history
                char1_profile = self.memory_system.character_profiles.get(char1)
                char2_profile = self.memory_system.character_profiles.get(char2)
                
                if not char1_profile or not char2_profile:
                    continue
                
                # Analyze relationship strength change
                prev_interactions = char1_profile.relationship_history.get(char2, [])
                
                if prev_interactions:
                    prev_emotional = np.mean([x[2] for x in prev_interactions[-3:]])  # Last 3 interactions
                    current_emotional = memory.emotional_weight
                    
                    emotional_change = current_emotional - prev_emotional
                    
                    if abs(emotional_change) > 0.5:  # Significant change
                        changes.append({
                            'character_pair': [char1, char2],
                            'change_type': 'emotional_shift',
                            'previous_weight': prev_emotional,
                            'new_weight': current_emotional,
                            'change_magnitude': emotional_change,
                            'chapter': memory.chapter,
                            'memory_id': memory.id
                        })
                
                # Detect relationship type changes
                prev_types = set(x[1] for x in prev_interactions[-5:])  # Recent types
                current_type = memory.relationship_type.value
                
                if current_type not in prev_types and prev_interactions:
                    changes.append({
                        'character_pair': [char1, char2],
                        'change_type': 'relationship_type_change',
                        'previous_types': list(prev_types),
                        'new_type': current_type,
                        'chapter': memory.chapter,
                        'memory_id': memory.id
                    })
        
        return changes


class NewCharacterIntegrator:
    """Enhanced new character integrator with LLM capabilities"""
    
    def __init__(self, memory_system: DynamicMemorySystem):
        self.memory_system = memory_system
    
    async def infer_character_traits(self, character_name: str):
        """Enhanced character trait inference"""
        if self.memory_system.llm_config.enabled:
            await self.memory_system._infer_character_traits_enhanced(character_name)
        else:
            self._infer_traits_basic(character_name)
    
    def _infer_traits_basic(self, character_name: str):
        """Basic trait inference without LLM"""
        char_memories = [
            m for m in self.memory_system.memories.values() 
            if m.character_perspective == character_name
        ]
        
        if not char_memories:
            return
        
        profile = self.memory_system.character_profiles[character_name]
        
        # Analyze emotional patterns
        emotional_weights = [m.emotional_weight for m in char_memories]
        avg_emotion = np.mean(emotional_weights)
        emotion_variance = np.var(emotional_weights)
        
        # Infer traits based on patterns
        traits = []
        
        if avg_emotion > 0.3:
            traits.append("optimistic")
        elif avg_emotion < -0.3:
            traits.append("pessimistic")
        
        if emotion_variance > 0.8:
            traits.append("emotionally_volatile")
        elif emotion_variance < 0.2:
            traits.append("emotionally_stable")
        
        # Analyze secrecy patterns
        secret_ratio = len([m for m in char_memories if m.is_secret]) / len(char_memories)
        
        if secret_ratio > 0.5:
            traits.append("secretive")
        elif secret_ratio < 0.1:
            traits.append("open")
        
        # Update profile
        profile.personality_traits = traits
        
        # Update core memory
        self.memory_system.core_memory[character_name]["personality"] = ", ".join(traits)
        
        logger.info(f"Inferred traits for {character_name}: {traits}")


# Keep all other classes from original file (MemoryUpdateAPI, create_flask_app, etc.)
# These remain unchanged but now work with the enhanced system

class MemoryUpdateAPI:
    """REST API for dynamic memory updates"""
    
    def __init__(self, dynamic_system: DynamicMemorySystem):
        self.dynamic_system = dynamic_system
    
    async def add_chapter_endpoint(self, chapter_data: Dict[str, Any]) -> Dict[str, Any]:
        """API endpoint for adding new chapters"""
        try:
            result = await self.dynamic_system.add_new_chapter(chapter_data)
            return {
                "status": "success",
                "data": result
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_character_evolution_endpoint(self, character_name: str) -> Dict[str, Any]:
        """API endpoint for character evolution data"""
        try:
            evolution = self.dynamic_system.get_character_evolution(character_name)
            return {
                "status": "success",
                "data": evolution
            }
        except Exception as e:
            return {
                "status": "error", 
                "message": str(e)
            }
    
    def get_growth_stats_endpoint(self) -> Dict[str, Any]:
        """API endpoint for narrative growth statistics"""
        try:
            stats = self.dynamic_system.get_narrative_growth_stats()
            return {
                "status": "success",
                "data": stats
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def search_memories_endpoint(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """API endpoint for memory search with dynamic filtering"""
        try:
            if filters is None:
                filters = {}
            
            memories = self.dynamic_system.retrieve_memories(
                query=query,
                character_perspective=filters.get('character'),
                relationship_type=filters.get('relationship_type'),
                top_k=filters.get('top_k', 10),
                chapter_range=filters.get('chapter_range')
            )
            
            memory_data = []
            for memory in memories:
                memory_data.append({
                    "id": memory.id,
                    "character": memory.character_perspective,
                    "content": memory.content,
                    "chapter": memory.chapter,
                    "participants": memory.participants,
                    "emotional_weight": memory.emotional_weight,
                    "importance": memory.importance,
                    "is_secret": memory.is_secret,
                    "memory_type": memory.memory_type.value,
                    "relationship_type": memory.relationship_type.value
                })
            
            return {
                "status": "success",
                "data": {
                    "memories": memory_data,
                    "total_found": len(memory_data)
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }


def create_flask_app(dynamic_system: DynamicMemorySystem):
    """Create Flask app with dynamic memory endpoints"""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    api = MemoryUpdateAPI(dynamic_system)
    
    @app.route('/api/chapters', methods=['POST'])
    def add_chapter():
        chapter_data = request.json
        # Convert to async call
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(api.add_chapter_endpoint(chapter_data))
        return jsonify(result)
    
    @app.route('/api/characters/<character_name>/evolution', methods=['GET'])
    def get_character_evolution(character_name):
        result = api.get_character_evolution_endpoint(character_name)
        return jsonify(result)
    
    @app.route('/api/stats/growth', methods=['GET'])
    def get_growth_stats():
        result = api.get_growth_stats_endpoint()
        return jsonify(result)
    
    @app.route('/api/memories/search', methods=['POST'])
    def search_memories():
        data = request.json
        query = data.get('query', '')
        filters = data.get('filters', {})
        result = api.search_memories_endpoint(query, filters)
        return jsonify(result)
    
    @app.route('/api/system/status', methods=['GET'])
    def system_status():
        stats = dynamic_system.get_narrative_growth_stats()
        return jsonify({
            "status": "active",
            "file_watcher_active": dynamic_system.file_watcher_active,
            "total_memories": stats['total_memories'],
            "total_characters": stats['total_characters'],
            "total_chapters": stats['total_chapters'],
            "recent_updates": stats['recent_updates'],
            "llm_enabled": dynamic_system.llm_config.enabled,
            "llm_model": dynamic_system.llm_config.model if dynamic_system.llm_config.enabled else None
        })
    
    return app


if __name__ == "__main__":
    # Example usage with LLM integration
    async def main():
        print("🚀 Enhanced Dynamic Memory System Example")
        
        # Initialize enhanced system
        dynamic_system = DynamicMemorySystem(
            openai_api_key="your-openai-api-key",  # Set your API key
            llm_model="gpt-4"  # or "gpt-3.5-turbo" for cost savings
        )
        
        # Example: Add new chapter with LLM enhancement
        new_chapter = {
            "chapter_number": 51,
            "synopsis": "A new consultant, Dr. Elena Vasquez, arrives and immediately recognizes Byleth's manipulation tactics through careful observation of office dynamics."
        }
        
        result = await dynamic_system.add_new_chapter(new_chapter)
        print(f"Enhanced chapter processing: {result}")
        
        # Example: Get enhanced character evolution
        if "Byleth" in dynamic_system.character_profiles:
            evolution = dynamic_system.get_character_evolution("Byleth")
            llm_analysis = evolution.get("llm_analysis", {})
            print(f"LLM Character Analysis for Byleth:")
            print(f"  Traits: {llm_analysis.get('personality_traits', [])}")
            print(f"  Emotional State: {llm_analysis.get('emotional_state', 'unknown')}")
            print(f"  Confidence: {llm_analysis.get('analysis_confidence', 0):.2f}")
    
    # Run the async example
    # asyncio.run(main())