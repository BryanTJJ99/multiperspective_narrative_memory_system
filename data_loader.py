# data_loader.py
"""
Enhanced Data Loader and Memory Extractor with LLM Integration
Intelligently parses chapter synopses to create character-specific memories
Now includes LLM-based content reframing for authentic character perspectives
"""

import json
import re
import asyncio
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sekai_memory_system import MemoryType, RelationshipType
from env_config import env_config

# OpenAI integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

@dataclass
class ExtractedMemory:
    character_perspective: str
    memory_type: MemoryType
    relationship_type: RelationshipType
    content: str
    participants: List[str]
    emotional_weight: float
    importance: float
    is_secret: bool
    tags: List[str]

class SekaiDataLoader:
    """Enhanced intelligent data loader for chapter data with LLM capabilities"""
    
    def __init__(self, openai_api_key: str = None, llm_model: str = None):
        self.character_names = [
            "Byleth", "Dimitri", "Sylvain", "Annette", "Dedue", 
            "Mercedes", "Felix", "Ashe"
        ]
        
        # LLM Configuration using environment settings
        self.openai_api_key = openai_api_key or env_config.openai_api_key
        self.llm_model = llm_model or env_config.openai_model
        self.temperature = env_config.openai_temperature
        self.max_tokens = env_config.openai_max_tokens
        
        if self.openai_api_key and OPENAI_AVAILABLE:
            openai.api_key = self.openai_api_key
            self.llm_enabled = True
        else:
            self.llm_enabled = False
        
        # Patterns for identifying different types of content
        self.secret_patterns = [
            r"secret", r"affair", r"hidden", r"concealed", r"covert",
            r"private", r"steamy", r"intimate", r"discrete", r"clandestine"
        ]
        
        self.emotion_patterns = {
            "positive": [r"happy", r"joy", r"excited", r"pleased", r"thrilled", r"satisfied"],
            "negative": [r"angry", r"furious", r"disappointed", r"hurt", r"betrayed", r"upset"],
            "neutral": [r"professional", r"casual", r"routine", r"standard"]
        }
        
        self.relationship_indicators = {
            "romantic": [r"kiss", r"passion", r"intimate", r"affair", r"attraction", r"chemistry"],
            "professional": [r"office", r"work", r"colleague", r"corporate", r"meeting"],
            "friendship": [r"friend", r"trust", r"support", r"loyalty", r"caring"]
        }
        
        # Character personality context for LLM reframing
        self.character_contexts = {
            "Byleth": {
                "personality_traits": ["manipulative", "strategic", "charming", "calculating"],
                "perspective_style": "analytical and self-serving",
                "emotional_default": "controlled manipulation",
                "secrets": ["affair with Dimitri", "affair with Sylvain"],
                "voice_style": "calculating internal monologue"
            },
            "Dimitri": {
                "personality_traits": ["intense", "passionate", "possessive", "loyal"],
                "perspective_style": "emotionally driven and focused",
                "emotional_default": "intense emotional responses",
                "secrets": ["affair with Byleth"],
                "voice_style": "passionate and sometimes obsessive"
            },
            "Sylvain": {
                "personality_traits": ["charming", "flirtatious", "conflicted", "charismatic"],
                "perspective_style": "socially aware but internally conflicted",
                "emotional_default": "charm masking inner turmoil",
                "secrets": ["affair with Byleth"],
                "voice_style": "smooth exterior with underlying tension"
            },
            "Annette": {
                "personality_traits": ["trusting", "optimistic", "caring", "gradually suspicious"],
                "perspective_style": "positive but increasingly observant",
                "emotional_default": "trusting optimism turning to doubt",
                "secrets": [],
                "voice_style": "warm but becoming more questioning"
            },
            "Dedue": {
                "personality_traits": ["observant", "protective", "methodical", "loyal"],
                "perspective_style": "protective analysis and careful observation",
                "emotional_default": "controlled concern for Dimitri",
                "secrets": ["knows about Byleth-Dimitri affair"],
                "voice_style": "careful, protective observations"
            }
        }
    
    def load_chapter_data(self, file_path: str) -> List[Dict]:
        """Load chapter data from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    async def extract_memories_from_chapter(self, chapter_data: Dict) -> List[ExtractedMemory]:
        """Extract multiple character perspectives from a single chapter with LLM enhancement"""
        chapter_num = chapter_data["chapter_number"]
        synopsis = chapter_data["synopsis"]
        
        memories = []
        
        # Identify characters mentioned in the synopsis
        mentioned_characters = self._identify_characters(synopsis)
        
        # Determine the primary perspective (usually the subject of the first sentence)
        primary_character = self._identify_primary_character(synopsis, mentioned_characters)
        
        # Extract basic memory components
        emotional_weight = self._calculate_emotional_weight(synopsis)
        importance = self._calculate_importance(synopsis, chapter_num)
        is_secret = self._is_secret_content(synopsis)
        memory_type = self._determine_memory_type(synopsis)
        relationship_type = self._determine_relationship_type(synopsis, mentioned_characters)
        tags = self._extract_tags(synopsis)
        
        # Create primary memory (from main character's perspective)
        if primary_character:
            primary_content = synopsis
            if self.llm_enabled:
                primary_content = await self._llm_reframe_content(
                    synopsis, primary_character, mentioned_characters, is_secret
                )
            
            memories.append(ExtractedMemory(
                character_perspective=primary_character,
                memory_type=memory_type,
                relationship_type=relationship_type,
                content=primary_content,
                participants=mentioned_characters,
                emotional_weight=emotional_weight,
                importance=importance,
                is_secret=is_secret,
                tags=tags
            ))
        
        # Create perspective-specific memories for other characters involved
        for character in mentioned_characters:
            if character != primary_character:
                char_emotional_weight = self._adjust_emotional_weight_for_character(
                    emotional_weight, character, synopsis
                )
                char_memory_type = self._adjust_memory_type_for_character(
                    memory_type, character, synopsis
                )
                
                # Enhanced content reframing with LLM
                if self.llm_enabled:
                    char_content = await self._llm_reframe_content(
                        synopsis, character, mentioned_characters, is_secret
                    )
                else:
                    char_content = self._reframe_content_for_character(synopsis, character)
                
                memories.append(ExtractedMemory(
                    character_perspective=character,
                    memory_type=char_memory_type,
                    relationship_type=relationship_type,
                    content=char_content,
                    participants=mentioned_characters,
                    emotional_weight=char_emotional_weight,
                    importance=importance * 0.8,  # Secondary perspective slightly less important
                    is_secret=is_secret,
                    tags=tags
                ))
        
        return memories
    
    async def _llm_reframe_content(self, original_content: str, character: str, 
                                  participants: List[str], is_secret: bool) -> str:
        """Use LLM to reframe content from a specific character's perspective"""
        if not self.llm_enabled:
            return self._reframe_content_for_character(original_content, character)
        
        char_context = self.character_contexts.get(character, {})
        
        prompt = f"""
Rewrite this narrative memory from {character}'s perspective:

Original: "{original_content}"

Character Context:
- Name: {character}
- Personality: {', '.join(char_context.get('personality_traits', []))}
- Perspective style: {char_context.get('perspective_style', 'observational')}
- Voice style: {char_context.get('voice_style', 'neutral')}
- Known secrets: {char_context.get('secrets', [])}
- Other characters present: {participants}
- This is a secret event: {is_secret}

Guidelines:
1. Write in {character}'s voice and internal perspective
2. Include their emotional interpretation and reaction
3. Show what they would notice, think, or feel
4. Reflect their personality in the language and focus
5. Consider their knowledge level about secrets
6. Keep the core events but show their unique viewpoint
7. Use first-person internal thoughts when appropriate

Return only the reframed memory text, staying true to the character.
"""
        
        try:
            # Updated for OpenAI v1.0+ API
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.openai_api_key)
            
            response = await client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a narrative perspective specialist who rewrites scenes from different character viewpoints."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            reframed_content = response.choices[0].message.content.strip()
            
            # Clean up any quotation marks that might wrap the response
            if reframed_content.startswith('"') and reframed_content.endswith('"'):
                reframed_content = reframed_content[1:-1]
            
            return reframed_content
            
        except Exception as e:
            print(f"LLM reframing failed for {character}: {e}")
            return self._reframe_content_for_character(original_content, character)
    
    def _identify_characters(self, text: str) -> List[str]:
        """Identify character names mentioned in the text"""
        mentioned = []
        for char_name in self.character_names:
            if char_name in text:
                mentioned.append(char_name)
        return mentioned
    
    def _identify_primary_character(self, text: str, mentioned_chars: List[str]) -> Optional[str]:
        """Identify the primary character (usually the subject of the first sentence)"""
        if not mentioned_chars:
            return None
        
        # Simple heuristic: first mentioned character is usually the primary one
        first_positions = {}
        for char in mentioned_chars:
            pos = text.find(char)
            if pos != -1:
                first_positions[char] = pos
        
        if first_positions:
            return min(first_positions.keys(), key=lambda x: first_positions[x])
        return mentioned_chars[0]
    
    def _calculate_emotional_weight(self, text: str) -> float:
        """Calculate emotional weight of the content (-1.0 to 1.0)"""
        positive_score = 0
        negative_score = 0
        
        text_lower = text.lower()
        
        for pattern in self.emotion_patterns["positive"]:
            positive_score += len(re.findall(pattern, text_lower))
        
        for pattern in self.emotion_patterns["negative"]:
            negative_score += len(re.findall(pattern, text_lower))
        
        # Normalize and return
        total_emotional = positive_score + negative_score
        if total_emotional == 0:
            return 0.0
        
        return (positive_score - negative_score) / total_emotional
    
    def _calculate_importance(self, text: str, chapter_num: int) -> float:
        """Calculate importance of the memory (0.0 to 1.0)"""
        importance = 0.5  # Base importance
        
        text_lower = text.lower()
        
        # Secret/affair content is highly important
        if self._is_secret_content(text):
            importance += 0.3
        
        # First encounters are important
        if chapter_num <= 10 and any(word in text_lower for word in ["first", "initial", "new", "steps into"]):
            importance += 0.2
        
        # Confrontations and discoveries are important
        if any(word in text_lower for word in ["confrontation", "discovers", "realizes", "catches"]):
            importance += 0.3
        
        # Evidence and observations are important
        if any(word in text_lower for word in ["evidence", "witness", "observes", "notices"]):
            importance += 0.25
        
        return min(1.0, importance)
    
    def _is_secret_content(self, text: str) -> bool:
        """Determine if content involves secrets"""
        text_lower = text.lower()
        for pattern in self.secret_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def _determine_memory_type(self, text: str) -> MemoryType:
        """Determine the type of memory based on content"""
        text_lower = text.lower()
        
        # Core memories: fundamental relationship changes
        if any(word in text_lower for word in ["affair", "relationship", "bond", "connection"]):
            return MemoryType.CORE
        
        # Procedural: how to do things, patterns of behavior
        if any(word in text_lower for word in ["approach", "strategy", "method", "technique"]):
            return MemoryType.PROCEDURAL
        
        # Semantic: facts and general knowledge
        if any(word in text_lower for word in ["knows", "understands", "aware", "realizes"]):
            return MemoryType.SEMANTIC
        
        # Resource: external information
        if any(word in text_lower for word in ["memo", "email", "message", "information"]):
            return MemoryType.RESOURCE
        
        # Default to episodic (specific events)
        return MemoryType.EPISODIC
    
    def _determine_relationship_type(self, text: str, participants: List[str]) -> RelationshipType:
        """Determine the relationship type based on content and participants"""
        # World memory: general events, company announcements, etc.
        if any(word in text.lower() for word in ["company", "office", "virus", "announcement", "general"]):
            return RelationshipType.WORLD_MEMORY
        
        # Character-to-User: direct interactions (assuming Byleth is the "user" character)
        if "Byleth" in participants and len(participants) == 2:
            return RelationshipType.CHARACTER_TO_USER
        
        # Inter-Character: multiple characters interacting
        return RelationshipType.INTER_CHARACTER
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags from the content"""
        tags = []
        text_lower = text.lower()
        
        # Location tags
        if "office" in text_lower:
            tags.append("office")
        if "restaurant" in text_lower:
            tags.append("restaurant")
        if "hotel" in text_lower:
            tags.append("hotel")
        if "home" in text_lower or "apartment" in text_lower:
            tags.append("private_space")
        
        # Event type tags
        if any(word in text_lower for word in ["meeting", "encounter", "conversation"]):
            tags.append("interaction")
        if any(word in text_lower for word in ["planning", "strategy", "scheme"]):
            tags.append("planning")
        if any(word in text_lower for word in ["discovery", "evidence", "caught"]):
            tags.append("revelation")
        
        # Relationship tags
        if any(word in text_lower for word in ["affair", "secret", "intimate"]):
            tags.append("secret_relationship")
        if any(word in text_lower for word in ["professional", "work", "colleague"]):
            tags.append("professional")
        
        return tags
    
    def _adjust_emotional_weight_for_character(self, base_weight: float, character: str, text: str) -> float:
        """Adjust emotional weight based on character's perspective"""
        # Character-specific emotional biases
        character_biases = {
            "Byleth": 0.1,      # Slightly more positive (manipulative confidence)
            "Dimitri": -0.1,    # Slightly more negative (intensity/pessimism)
            "Sylvain": 0.0,     # Neutral
            "Annette": 0.3,     # Very positive (optimistic)
            "Dedue": -0.2,      # More negative (cautious)
            "Mercedes": 0.2,    # Positive (caring)
            "Felix": -0.1,      # Slightly negative (cynical)
            "Ashe": 0.1         # Slightly positive (hopeful)
        }
        
        bias = character_biases.get(character, 0.0)
        adjusted_weight = base_weight + bias
        
        # Specific adjustments based on content and character
        text_lower = text.lower()
        
        if character == "Annette":
            # Annette doesn't know about the affairs, so betrayal scenes are less negative for her
            if "betrayal" in text_lower and "Sylvain" in text:
                adjusted_weight = max(adjusted_weight, -0.2)
        
        elif character == "Dedue":
            # Dedue is protective, so threats to Dimitri are very negative
            if "Dimitri" in text and any(word in text_lower for word in ["threat", "danger", "risk"]):
                adjusted_weight -= 0.3
        
        elif character == "Byleth":
            # Byleth sees manipulation as success
            if any(word in text_lower for word in ["manipulation", "control", "strategy"]):
                adjusted_weight += 0.2
        
        return max(-1.0, min(1.0, adjusted_weight))
    
    def _adjust_memory_type_for_character(self, base_type: MemoryType, character: str, text: str) -> MemoryType:
        """Adjust memory type based on character's perspective"""
        
        # Dedue tends to store observations as procedural (how to protect Dimitri)
        if character == "Dedue" and "observes" in text.lower():
            return MemoryType.PROCEDURAL
        
        # Byleth stores strategic information as procedural
        if character == "Byleth" and any(word in text.lower() for word in ["strategy", "plan", "manipulation"]):
            return MemoryType.PROCEDURAL
        
        # Annette stores relationship information as core memories
        if character == "Annette" and "Sylvain" in text:
            return MemoryType.CORE
        
        return base_type
    
    def _reframe_content_for_character(self, original_content: str, character: str) -> str:
        """Reframe content from a specific character's perspective (fallback method)"""
        # This is a simplified version for when LLM is not available
        
        perspective_frames = {
            "Dedue": f"[Observing protectively] {original_content}",
            "Annette": f"[Unaware of deceptions] {original_content}",
            "Felix": f"[Analytical assessment] {original_content}",
            "Mercedes": f"[Caring observation] {original_content}",
            "Dimitri": f"[Intense focus] {original_content}",
            "Sylvain": f"[Charming exterior] {original_content}"
        }
        
        return perspective_frames.get(character, original_content)


class MemoryDataProcessor:
    """Enhanced processor for loading and converting data into memory system"""
    
    def __init__(self, memory_system, openai_api_key: str = None):
        self.memory_system = memory_system
        self.data_loader = SekaiDataLoader(openai_api_key)
    
    async def process_json_file(self, file_path: str) -> Dict[str, int]:
        """Process the memory_data.json file and load into memory system"""
        
        # Load chapter data
        chapter_data = self.data_loader.load_chapter_data(file_path)
        
        stats = {
            "chapters_processed": 0,
            "memories_created": 0,
            "characters_involved": set(),
            "llm_enhanced": self.data_loader.llm_enabled
        }
        
        # Process each chapter
        for chapter_info in chapter_data:
            extracted_memories = await self.data_loader.extract_memories_from_chapter(chapter_info)
            
            # Add each extracted memory to the system
            for memory_data in extracted_memories:
                memory_id = self.memory_system.add_memory(
                    character_perspective=memory_data.character_perspective,
                    memory_type=memory_data.memory_type,
                    relationship_type=memory_data.relationship_type,
                    content=memory_data.content,
                    chapter=chapter_info["chapter_number"],
                    participants=memory_data.participants,
                    emotional_weight=memory_data.emotional_weight,
                    importance=memory_data.importance,
                    is_secret=memory_data.is_secret
                )
                
                stats["memories_created"] += 1
                stats["characters_involved"].add(memory_data.character_perspective)
            
            stats["chapters_processed"] += 1
        
        # Convert set to list for JSON serialization
        stats["characters_involved"] = list(stats["characters_involved"])
        
        return stats


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize memory system
        from dynamic_memory_system import DynamicMemorySystem
        memory_system = DynamicMemorySystem(openai_api_key="your-openai-api-key")
        
        # Initialize enhanced processor
        processor = MemoryDataProcessor(memory_system, openai_api_key="your-openai-api-key")
        
        # Process the data file
        stats = await processor.process_json_file("memory_data.json")
        
        print(f"Enhanced processing complete!")
        print(f"Chapters processed: {stats['chapters_processed']}")
        print(f"Memories created: {stats['memories_created']}")
        print(f"Characters involved: {stats['characters_involved']}")
        print(f"LLM enhanced: {stats['llm_enhanced']}")
        
        # Test some queries
        print("\n=== Testing Enhanced Memory Retrieval ===")
        
        # Test 1: Byleth's relationship with Dimitri
        byleth_dimitri_memories = memory_system.retrieve_memories(
            "Byleth Dimitri relationship affair", 
            character_perspective="Byleth",
            top_k=3
        )
        print(f"\nByleth's memories about Dimitri: {len(byleth_dimitri_memories)} found")
        for memory in byleth_dimitri_memories:
            print(f"  Chapter {memory.chapter}: {memory.content[:100]}...")
        
        # Test 2: Dedue's observations
        dedue_memories = memory_system.retrieve_memories(
            "observation evidence discovery",
            character_perspective="Dedue",
            top_k=3
        )
        print(f"\nDedue's observations: {len(dedue_memories)} found")
        for memory in dedue_memories:
            print(f"  Chapter {memory.chapter}: {memory.content[:100]}...")
    
    # Run the async example
    # asyncio.run(main())