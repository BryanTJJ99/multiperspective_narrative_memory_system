"""
Comprehensive Memory Evaluation Pipeline
Answers the three key evaluation questions:
A) Does the system retrieve the right memories?
B) Are memories internally consistent across time, characters, and world state?
C) Additional evaluation metrics for system performance
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import networkx as nx

from src.sekai_memory_system import SekaiMemorySystem, Memory, MemoryType, RelationshipType

@dataclass
class EvaluationResult:
    metric_name: str
    score: float
    details: Dict[str, Any]
    timestamp: datetime
    test_case_id: Optional[str] = None

@dataclass  
class TestCase:
    id: str
    query: str
    expected_memory_ids: List[str]
    expected_characters: List[str]
    chapter_range: Optional[Tuple[int, int]]
    description: str

class MemoryRetrievalEvaluator:
    """Evaluates precision and recall of memory retrieval (Question A)"""
    
    def __init__(self, memory_system: SekaiMemorySystem):
        self.memory_system = memory_system
        
    def create_test_cases(self) -> List[TestCase]:
        """Create test cases based on known story patterns"""
        
        # Get all memories for reference
        all_memories = list(self.memory_system.memories.values())
        
        test_cases = []
        
        # Test Case 1: Byleth-Dimitri affair memories
        byleth_dimitri_memories = [
            m.id for m in all_memories 
            if "Byleth" in m.participants and "Dimitri" in m.participants 
            and (m.is_secret or "affair" in m.content.lower())
        ]
        
        test_cases.append(TestCase(
            id="byleth_dimitri_affair",
            query="Byleth Dimitri secret relationship affair",
            expected_memory_ids=byleth_dimitri_memories,
            expected_characters=["Byleth", "Dimitri"],
            chapter_range=(3, 30),
            description="Memories about Byleth and Dimitri's secret affair"
        ))
        
        # Test Case 2: Dedue's observations and discoveries
        dedue_memories = [
            m.id for m in all_memories
            if m.character_perspective == "Dedue" 
            and any(word in m.content.lower() for word in ["observes", "evidence", "notices", "witness"])
        ]
        
        test_cases.append(TestCase(
            id="dedue_observations",
            query="Dedue observes evidence discovery witness",
            expected_memory_ids=dedue_memories,
            expected_characters=["Dedue"],
            chapter_range=(9, 35),
            description="Dedue's observations and evidence gathering"
        ))
        
        # Test Case 3: Sylvain-Annette relationship
        sylvain_annette_memories = [
            m.id for m in all_memories
            if "Sylvain" in m.participants and "Annette" in m.participants
            and not m.is_secret  # Their official relationship
        ]
        
        test_cases.append(TestCase(
            id="sylvain_annette_relationship",
            query="Sylvain Annette relationship official couple",
            expected_memory_ids=sylvain_annette_memories,
            expected_characters=["Sylvain", "Annette"],
            chapter_range=(1, 50),
            description="Sylvain and Annette's official relationship"
        ))
        
        # Test Case 4: Office/workplace memories
        office_memories = [
            m.id for m in all_memories
            if any(word in m.content.lower() for word in ["office", "work", "corporate", "company", "professional"])
        ]
        
        test_cases.append(TestCase(
            id="office_workplace",
            query="office work corporate professional workplace",
            expected_memory_ids=office_memories,
            expected_characters=["Byleth", "Dimitri", "Sylvain", "Annette"],
            chapter_range=(1, 50),
            description="Workplace and office-related memories"
        ))
        
        # Test Case 5: Secret/deception memories
        secret_memories = [
            m.id for m in all_memories
            if m.is_secret or any(word in m.content.lower() for word in ["secret", "lie", "deception", "manipulation"])
        ]
        
        test_cases.append(TestCase(
            id="secrets_deception",
            query="secret deception manipulation lie hidden",
            expected_memory_ids=secret_memories,
            expected_characters=["Byleth"],
            chapter_range=(1, 50),
            description="Memories involving secrets and deception"
        ))
        
        return test_cases
    
    def evaluate_retrieval_accuracy(self, test_cases: List[TestCase]) -> List[EvaluationResult]:
        """Evaluate precision, recall, and F1 score for memory retrieval"""
        results = []
        
        for test_case in test_cases:
            # Retrieve memories using the system
            retrieved_memories = self.memory_system.retrieve_memories(
                query=test_case.query,
                top_k=20  # Retrieve more to test ranking
            )
            
            retrieved_ids = set(m.id for m in retrieved_memories)
            expected_ids = set(test_case.expected_memory_ids)
            
            # Calculate metrics
            true_positives = len(retrieved_ids & expected_ids)
            false_positives = len(retrieved_ids - expected_ids)
            false_negatives = len(expected_ids - retrieved_ids)
            
            precision = true_positives / len(retrieved_ids) if retrieved_ids else 0.0
            recall = true_positives / len(expected_ids) if expected_ids else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Create result
            result = EvaluationResult(
                metric_name="retrieval_accuracy",
                score=f1_score,
                details={
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                    "retrieved_count": len(retrieved_ids),
                    "expected_count": len(expected_ids)
                },
                timestamp=datetime.now(),
                test_case_id=test_case.id
            )
            
            results.append(result)
        
        return results

class MemoryConsistencyEvaluator:
    """Evaluates memory consistency across time, characters, and world state (Question B)"""
    
    def __init__(self, memory_system: SekaiMemorySystem):
        self.memory_system = memory_system
    
    def evaluate_temporal_consistency(self) -> EvaluationResult:
        """Check if memories maintain consistency over time"""
        
        all_memories = list(self.memory_system.memories.values())
        consistency_issues = []
        
        # Group memories by character
        character_memories = defaultdict(list)
        for memory in all_memories:
            character_memories[memory.character_perspective].append(memory)
        
        # Check each character's memories for consistency
        total_inconsistencies = 0
        total_comparisons = 0
        
        for character, memories in character_memories.items():
            memories.sort(key=lambda x: x.chapter)
            
            for i in range(len(memories) - 1):
                for j in range(i + 1, min(i + 5, len(memories))):  # Check nearby memories
                    memory1, memory2 = memories[i], memories[j]
                    
                    # Check for contradictory emotional weights about same people
                    shared_participants = set(memory1.participants) & set(memory2.participants)
                    if len(shared_participants) >= 2:  # Same relationship context
                        total_comparisons += 1
                        
                        # Large emotional weight differences might indicate inconsistency
                        weight_diff = abs(memory1.emotional_weight - memory2.emotional_weight)
                        if weight_diff > 1.5:  # Threshold for inconsistency
                            total_inconsistencies += 1
                            consistency_issues.append({
                                "character": character,
                                "memory1_chapter": memory1.chapter,
                                "memory2_chapter": memory2.chapter,
                                "weight_difference": weight_diff,
                                "shared_participants": list(shared_participants)
                            })
        
        consistency_score = 1.0 - (total_inconsistencies / max(1, total_comparisons))
        
        return EvaluationResult(
            metric_name="temporal_consistency",
            score=consistency_score,
            details={
                "consistency_score": consistency_score,
                "total_inconsistencies": total_inconsistencies,
                "total_comparisons": total_comparisons,
                "consistency_issues": consistency_issues
            },
            timestamp=datetime.now()
        )
    
    def evaluate_cross_character_consistency(self) -> EvaluationResult:
        """Check if shared events are consistently remembered across characters"""
        
        all_memories = list(self.memory_system.memories.values())
        shared_events = defaultdict(list)
        
        # Group memories by shared participants (potential shared events)
        for memory in all_memories:
            if len(memory.participants) >= 2:
                # Create a key for shared events
                participants_key = tuple(sorted(memory.participants))
                shared_events[participants_key].append(memory)
        
        inconsistency_count = 0
        total_shared_events = 0
        consistency_details = []
        
        for participants, memories in shared_events.items():
            if len(memories) >= 2:  # Multiple perspectives on same event
                total_shared_events += 1
                
                # Check if memories are from similar chapters (same event)
                memories.sort(key=lambda x: x.chapter)
                
                # Group by chapter proximity (within 2 chapters = same event)
                event_groups = []
                current_group = [memories[0]]
                
                for memory in memories[1:]:
                    if memory.chapter - current_group[-1].chapter <= 2:
                        current_group.append(memory)
                    else:
                        event_groups.append(current_group)
                        current_group = [memory]
                event_groups.append(current_group)
                
                # Check consistency within each event group
                for group in event_groups:
                    if len(group) >= 2:
                        # Check if different characters remember the event very differently
                        emotional_weights = [m.emotional_weight for m in group]
                        weight_variance = np.var(emotional_weights)
                        
                        # High variance in emotional weights might indicate inconsistency
                        if weight_variance > 0.8:  # Threshold for inconsistency
                            inconsistency_count += 1
                            consistency_details.append({
                                "participants": list(participants),
                                "chapter_range": [min(m.chapter for m in group), max(m.chapter for m in group)],
                                "perspectives": [m.character_perspective for m in group],
                                "emotional_variance": weight_variance,
                                "memories": [{"char": m.character_perspective, "weight": m.emotional_weight} for m in group]
                            })
        
        consistency_score = 1.0 - (inconsistency_count / max(1, total_shared_events))
        
        return EvaluationResult(
            metric_name="cross_character_consistency",
            score=consistency_score,
            details={
                "consistency_score": consistency_score,
                "inconsistency_count": inconsistency_count,
                "total_shared_events": total_shared_events,
                "consistency_details": consistency_details
            },
            timestamp=datetime.now()
        )
    
    def evaluate_world_state_consistency(self) -> EvaluationResult:
        """Check if world state changes are consistently reflected across all characters"""
        
        # Get world memory and general environmental memories
        world_memories = [
            m for m in self.memory_system.memories.values()
            if m.relationship_type == RelationshipType.WORLD_MEMORY or 
            any(word in m.content.lower() for word in ["company", "office", "virus", "announcement"])
        ]
        
        world_memories.sort(key=lambda x: x.chapter)
        
        # Check if major world events are reflected in character memories
        major_world_events = []
        for memory in world_memories:
            if any(word in memory.content.lower() for word in ["virus", "crisis", "announcement", "outbreak"]):
                major_world_events.append(memory)
        
        consistency_issues = 0
        total_checks = 0
        
        for world_event in major_world_events:
            # Check if characters have corresponding memories around the same time
            event_chapter = world_event.chapter
            
            character_awareness = {}
            for character in self.memory_system.core_memory.keys():
                char_memories_around_event = [
                    m for m in self.memory_system.memories.values()
                    if (m.character_perspective == character and
                        abs(m.chapter - event_chapter) <= 3)  # Within 3 chapters
                ]
                
                # Check if character has any memory mentioning the world event
                aware = any(
                    any(keyword in m.content.lower() for keyword in ["virus", "crisis", "health"])
                    for m in char_memories_around_event
                )
                character_awareness[character] = aware
                total_checks += 1
            
            # If major world event but some characters have no awareness, that's inconsistent
            awareness_rate = sum(character_awareness.values()) / len(character_awareness)
            if awareness_rate < 0.3:  # Less than 30% of characters aware
                consistency_issues += 1
        
        consistency_score = 1.0 - (consistency_issues / max(1, len(major_world_events)))
        
        return EvaluationResult(
            metric_name="world_state_consistency",
            score=consistency_score,
            details={
                "consistency_score": consistency_score,
                "major_world_events": len(major_world_events),
                "consistency_issues": consistency_issues,
                "world_events_detected": [m.content[:100] for m in major_world_events]
            },
            timestamp=datetime.now()
        )

class AdditionalMetricsEvaluator:
    """Additional evaluation metrics (Question C)"""
    
    def __init__(self, memory_system: SekaiMemorySystem):
        self.memory_system = memory_system
    
    def evaluate_character_distinctiveness(self) -> EvaluationResult:
        """Evaluate how well the system maintains distinct character personalities"""
        
        character_profiles = {}
        
        for character in self.memory_system.core_memory.keys():
            char_memories = [
                m for m in self.memory_system.memories.values()
                if m.character_perspective == character
            ]
            
            if char_memories:
                # Calculate character profile based on memories
                avg_emotional_weight = np.mean([m.emotional_weight for m in char_memories])
                secret_ratio = sum(1 for m in char_memories if m.is_secret) / len(char_memories)
                avg_importance = np.mean([m.importance for m in char_memories])
                memory_types = Counter(m.memory_type for m in char_memories)
                
                character_profiles[character] = {
                    "avg_emotional_weight": avg_emotional_weight,
                    "secret_ratio": secret_ratio,
                    "avg_importance": avg_importance,
                    "dominant_memory_type": memory_types.most_common(1)[0][0].value if memory_types else "none",
                    "memory_count": len(char_memories)
                }
        
        # Calculate distinctiveness by comparing character profiles
        distinctiveness_scores = []
        characters = list(character_profiles.keys())
        
        for i, char1 in enumerate(characters):
            for char2 in characters[i+1:]:
                profile1 = character_profiles[char1]
                profile2 = character_profiles[char2]
                
                # Calculate profile similarity (lower = more distinct)
                weight_diff = abs(profile1["avg_emotional_weight"] - profile2["avg_emotional_weight"])
                secret_diff = abs(profile1["secret_ratio"] - profile2["secret_ratio"])
                importance_diff = abs(profile1["avg_importance"] - profile2["avg_importance"])
                
                similarity = 1.0 - (weight_diff + secret_diff + importance_diff) / 3.0
                distinctiveness = 1.0 - similarity
                distinctiveness_scores.append(distinctiveness)
        
        avg_distinctiveness = np.mean(distinctiveness_scores) if distinctiveness_scores else 0.0
        
        return EvaluationResult(
            metric_name="character_distinctiveness",
            score=avg_distinctiveness,
            details={
                "distinctiveness_score": avg_distinctiveness,
                "character_profiles": character_profiles,
                "pairwise_distinctiveness": distinctiveness_scores
            },
            timestamp=datetime.now()
        )
    
    def evaluate_secret_management(self) -> EvaluationResult:
        """Evaluate how well the system manages secrets and information asymmetry"""
        
        all_memories = list(self.memory_system.memories.values())
        secret_memories = [m for m in all_memories if m.is_secret]
        
        # Check if secrets are properly compartmentalized
        secret_leakage_count = 0
        total_secret_checks = 0
        
        for secret_memory in secret_memories:
            # Secrets should not be known by characters who shouldn't know them
            secret_participants = set(secret_memory.participants)
            
            # Check if characters outside the secret have memories about it
            for character in self.memory_system.core_memory.keys():
                if character not in secret_participants:
                    # This character shouldn't know the secret
                    char_memories = [
                        m for m in all_memories 
                        if m.character_perspective == character
                    ]
                    
                    # Check if they have memories that reveal the secret
                    for char_memory in char_memories:
                        if (char_memory.is_secret and 
                            len(set(char_memory.participants) & secret_participants) >= 2):
                            # This character has secret knowledge they shouldn't have
                            secret_leakage_count += 1
                            break
                    
                    total_secret_checks += 1
        
        secret_containment_score = 1.0 - (secret_leakage_count / max(1, total_secret_checks))
        
        # Analyze secret distribution
        secret_by_character = defaultdict(int)
        for memory in secret_memories:
            secret_by_character[memory.character_perspective] += 1
        
        return EvaluationResult(
            metric_name="secret_management",
            score=secret_containment_score,
            details={
                "secret_containment_score": secret_containment_score,
                "total_secrets": len(secret_memories),
                "secret_leakage_count": secret_leakage_count,
                "secrets_by_character": dict(secret_by_character)
            },
            timestamp=datetime.now()
        )
    
    def evaluate_relationship_tracking(self) -> EvaluationResult:
        """Evaluate how well the system tracks evolving relationships"""
        
        # Get relationship timeline for key pairs
        key_relationships = [
            ("Byleth", "Dimitri"),
            ("Byleth", "Sylvain"), 
            ("Sylvain", "Annette"),
            ("Dimitri", "Dedue")
        ]
        
        relationship_scores = []
        relationship_details = {}
        
        for char1, char2 in key_relationships:
            timeline = self.memory_system.get_character_relationship_timeline(char1, char2)
            
            if len(timeline) >= 2:
                # Check if relationship shows realistic progression
                emotional_progression = [m.emotional_weight for m in timeline]
                
                # Calculate relationship development score
                # Good relationships should show some progression/change over time
                progression_variance = np.var(emotional_progression)
                
                # But not too erratic (consistency within development)
                smoothness_score = 1.0 / (1.0 + progression_variance) if progression_variance > 0 else 0.5
                
                # Check for realistic relationship milestones
                milestone_score = 0.0
                for i in range(1, len(timeline)):
                    prev_memory = timeline[i-1]
                    curr_memory = timeline[i]
                    
                    # Positive progression indicators
                    if curr_memory.importance > prev_memory.importance:
                        milestone_score += 0.2
                    if curr_memory.chapter - prev_memory.chapter <= 10:  # Regular interaction
                        milestone_score += 0.1
                
                milestone_score = min(1.0, milestone_score)
                
                relationship_score = (smoothness_score + milestone_score) / 2.0
                relationship_scores.append(relationship_score)
                
                relationship_details[f"{char1}-{char2}"] = {
                    "timeline_length": len(timeline),
                    "progression_variance": progression_variance,
                    "smoothness_score": smoothness_score,
                    "milestone_score": milestone_score,
                    "relationship_score": relationship_score,
                    "emotional_progression": emotional_progression
                }
        
        avg_relationship_score = np.mean(relationship_scores) if relationship_scores else 0.0
        
        return EvaluationResult(
            metric_name="relationship_tracking",
            score=avg_relationship_score,
            details={
                "avg_relationship_score": avg_relationship_score,
                "relationships_analyzed": len(key_relationships),
                "relationship_details": relationship_details
            },
            timestamp=datetime.now()
        )

class ComprehensiveEvaluationPipeline:
    """Main evaluation pipeline that orchestrates all evaluations"""
    
    def __init__(self, memory_system: SekaiMemorySystem):
        self.memory_system = memory_system
        self.retrieval_evaluator = MemoryRetrievalEvaluator(memory_system)
        self.consistency_evaluator = MemoryConsistencyEvaluator(memory_system)
        self.additional_evaluator = AdditionalMetricsEvaluator(memory_system)
        
        self.evaluation_history = []
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run all evaluations and return comprehensive results"""
        
        print("🔍 Running Comprehensive Memory System Evaluation...")
        results = {
            "timestamp": datetime.now().isoformat(),
            "system_stats": self._get_system_stats(),
            "question_a_retrieval": {},
            "question_b_consistency": {},
            "question_c_additional": {},
            "overall_score": 0.0
        }
        
        # Question A: Retrieval Accuracy
        print("📊 Evaluating retrieval accuracy (Question A)...")
        test_cases = self.retrieval_evaluator.create_test_cases()
        retrieval_results = self.retrieval_evaluator.evaluate_retrieval_accuracy(test_cases)
        
        # Aggregate retrieval results
        retrieval_scores = [r.score for r in retrieval_results]
        results["question_a_retrieval"] = {
            "overall_score": np.mean(retrieval_scores),
            "test_cases": {r.test_case_id: r.details for r in retrieval_results},
            "average_precision": np.mean([r.details["precision"] for r in retrieval_results]),
            "average_recall": np.mean([r.details["recall"] for r in retrieval_results]),
            "average_f1": np.mean([r.details["f1_score"] for r in retrieval_results])
        }
        
        # Question B: Consistency
        print("🔄 Evaluating memory consistency (Question B)...")
        temporal_consistency = self.consistency_evaluator.evaluate_temporal_consistency()
        cross_char_consistency = self.consistency_evaluator.evaluate_cross_character_consistency()
        world_state_consistency = self.consistency_evaluator.evaluate_world_state_consistency()
        
        results["question_b_consistency"] = {
            "overall_score": np.mean([
                temporal_consistency.score,
                cross_char_consistency.score,
                world_state_consistency.score
            ]),
            "temporal_consistency": temporal_consistency.details,
            "cross_character_consistency": cross_char_consistency.details,
            "world_state_consistency": world_state_consistency.details
        }
        
        # Question C: Additional Metrics
        print("⚡ Evaluating additional metrics (Question C)...")
        char_distinctiveness = self.additional_evaluator.evaluate_character_distinctiveness()
        secret_management = self.additional_evaluator.evaluate_secret_management()
        relationship_tracking = self.additional_evaluator.evaluate_relationship_tracking()
        
        results["question_c_additional"] = {
            "overall_score": np.mean([
                char_distinctiveness.score,
                secret_management.score,
                relationship_tracking.score
            ]),
            "character_distinctiveness": char_distinctiveness.details,
            "secret_management": secret_management.details,
            "relationship_tracking": relationship_tracking.details
        }
        
        # Calculate overall system score
        results["overall_score"] = np.mean([
            results["question_a_retrieval"]["overall_score"],
            results["question_b_consistency"]["overall_score"],
            results["question_c_additional"]["overall_score"]
        ])
        
        # Store in history
        self.evaluation_history.append(results)
        
        print(f"✅ Evaluation complete! Overall score: {results['overall_score']:.3f}")
        return results
    
    def _get_system_stats(self) -> Dict[str, Any]:
        """Get basic system statistics"""
        all_memories = list(self.memory_system.memories.values())
        
        return {
            "total_memories": len(all_memories),
            "memories_by_type": {
                memory_type.value: len([m for m in all_memories if m.memory_type == memory_type])
                for memory_type in MemoryType
            },
            "memories_by_relationship": {
                rel_type.value: len([m for m in all_memories if m.relationship_type == rel_type])
                for rel_type in RelationshipType
            },
            "memories_by_character": {
                char: len([m for m in all_memories if m.character_perspective == char])
                for char in self.memory_system.core_memory.keys()
            },
            "secret_memories": len([m for m in all_memories if m.is_secret]),
            "chapter_range": [min(m.chapter for m in all_memories), max(m.chapter for m in all_memories)] if all_memories else [0, 0]
        }
    
    def generate_evaluation_report(self, results: Dict[str, Any], output_file: str = "evaluation_report.html"):
        """Generate a comprehensive HTML evaluation report"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sekai Memory System Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 20px; margin-bottom: 30px; }}
                .score-card {{ background-color: #ecf0f1; padding: 20px; margin: 20px 0; border-radius: 8px; text-align: center; }}
                .score {{ font-size: 48px; font-weight: bold; color: #2c3e50; }}
                .score-label {{ font-size: 18px; color: #7f8c8d; margin-top: 10px; }}
                .section {{ margin: 30px 0; }}
                .section h2 {{ color: #2c3e50; border-left: 4px solid #3498db; padding-left: 15px; }}
                .metric {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #27ae60; }}
                .metric h4 {{ margin: 0 0 10px 0; color: #2c3e50; }}
                .metric-score {{ font-weight: bold; font-size: 18px; color: #27ae60; }}
                .details {{ margin-top: 10px; font-size: 14px; color: #555; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-box {{ background-color: #3498db; color: white; padding: 20px; border-radius: 8px; text-align: center; }}
                .stat-number {{ font-size: 24px; font-weight: bold; }}
                .stat-label {{ margin-top: 5px; opacity: 0.9; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🧠 Sekai Memory System Evaluation Report</h1>
                    <p>Generated on {results['timestamp']}</p>
                </div>
                
                <div class="score-card">
                    <div class="score">{results['overall_score']:.3f}</div>
                    <div class="score-label">Overall System Score</div>
                </div>
                
                <div class="section">
                    <h2>📊 System Statistics</h2>
                    <div class="stats-grid">
                        <div class="stat-box">
                            <div class="stat-number">{results['system_stats']['total_memories']}</div>
                            <div class="stat-label">Total Memories</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-number">{results['system_stats']['secret_memories']}</div>
                            <div class="stat-label">Secret Memories</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-number">{results['system_stats']['chapter_range'][1] - results['system_stats']['chapter_range'][0] + 1}</div>
                            <div class="stat-label">Chapters Covered</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-number">{len(results['system_stats']['memories_by_character'])}</div>
                            <div class="stat-label">Characters</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎯 Question A: Retrieval Accuracy</h2>
                    <div class="metric">
                        <h4>Overall Retrieval Score</h4>
                        <div class="metric-score">{results['question_a_retrieval']['overall_score']:.3f}</div>
                        <div class="details">
                            <strong>Precision:</strong> {results['question_a_retrieval']['average_precision']:.3f} | 
                            <strong>Recall:</strong> {results['question_a_retrieval']['average_recall']:.3f} | 
                            <strong>F1 Score:</strong> {results['question_a_retrieval']['average_f1']:.3f}
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔄 Question B: Memory Consistency</h2>
                    <div class="metric">
                        <h4>Overall Consistency Score</h4>
                        <div class="metric-score">{results['question_b_consistency']['overall_score']:.3f}</div>
                    </div>
                    
                    <div class="metric">
                        <h4>Temporal Consistency</h4>
                        <div class="metric-score">{results['question_b_consistency']['temporal_consistency']['consistency_score']:.3f}</div>
                        <div class="details">
                            {results['question_b_consistency']['temporal_consistency']['total_inconsistencies']} inconsistencies found 
                            out of {results['question_b_consistency']['temporal_consistency']['total_comparisons']} comparisons
                        </div>
                    </div>
                    
                    <div class="metric">
                        <h4>Cross-Character Consistency</h4>
                        <div class="metric-score">{results['question_b_consistency']['cross_character_consistency']['consistency_score']:.3f}</div>
                        <div class="details">
                            {results['question_b_consistency']['cross_character_consistency']['inconsistency_count']} inconsistencies 
                            across {results['question_b_consistency']['cross_character_consistency']['total_shared_events']} shared events
                        </div>
                    </div>
                    
                    <div class="metric">
                        <h4>World State Consistency</h4>
                        <div class="metric-score">{results['question_b_consistency']['world_state_consistency']['consistency_score']:.3f}</div>
                        <div class="details">
                            {results['question_b_consistency']['world_state_consistency']['major_world_events']} major world events detected
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>⚡ Question C: Additional Metrics</h2>
                    <div class="metric">
                        <h4>Overall Additional Score</h4>
                        <div class="metric-score">{results['question_c_additional']['overall_score']:.3f}</div>
                    </div>
                    
                    <div class="metric">
                        <h4>Character Distinctiveness</h4>
                        <div class="metric-score">{results['question_c_additional']['character_distinctiveness']['distinctiveness_score']:.3f}</div>
                        <div class="details">Measures how well characters maintain unique personality profiles</div>
                    </div>
                    
                    <div class="metric">
                        <h4>Secret Management</h4>
                        <div class="metric-score">{results['question_c_additional']['secret_management']['secret_containment_score']:.3f}</div>
                        <div class="details">
                            {results['question_c_additional']['secret_management']['total_secrets']} secrets tracked, 
                            {results['question_c_additional']['secret_management']['secret_leakage_count']} potential leakages
                        </div>
                    </div>
                    
                    <div class="metric">
                        <h4>Relationship Tracking</h4>
                        <div class="metric-score">{results['question_c_additional']['relationship_tracking']['avg_relationship_score']:.3f}</div>
                        <div class="details">
                            {results['question_c_additional']['relationship_tracking']['relationships_analyzed']} key relationships analyzed
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📈 Recommendations</h2>
                    <div class="details">
                        <h4>Areas for Improvement:</h4>
                        <ul>
        """
        
        # Add recommendations based on scores
        if results['question_a_retrieval']['overall_score'] < 0.7:
            html_content += "<li>Consider improving memory retrieval algorithms or embedding quality</li>"
        
        if results['question_b_consistency']['overall_score'] < 0.7:
            html_content += "<li>Review memory consistency mechanisms and contradiction detection</li>"
        
        if results['question_c_additional']['character_distinctiveness']['distinctiveness_score'] < 0.6:
            html_content += "<li>Enhance character-specific memory processing to maintain distinct personalities</li>"
        
        if results['question_c_additional']['secret_management']['secret_containment_score'] < 0.8:
            html_content += "<li>Strengthen secret compartmentalization and access control</li>"
        
        html_content += """
                        </ul>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 Evaluation report saved to {output_file}")
    
    def run_online_evaluation(self, new_memories: List[str]) -> Dict[str, float]:
        """
        Online evaluation for production monitoring
        Lightweight version that runs on new memory additions
        """
        
        online_metrics = {}
        
        # Quick consistency check for new memories
        if new_memories:
            recent_memories = [self.memory_system.memories[mid] for mid in new_memories if mid in self.memory_system.memories]
            
            # Check for immediate contradictions
            contradiction_count = 0
            for memory in recent_memories:
                contradictions = self.memory_system.detect_contradictions(memory.id)
                contradiction_count += len(contradictions)
            
            online_metrics['immediate_consistency'] = 1.0 - min(1.0, contradiction_count / len(recent_memories))
            
            # Check secret leakage
            secret_leakage = 0
            for memory in recent_memories:
                if memory.is_secret:
                    # Quick check: are there non-participants with similar memories?
                    similar_memories = self.memory_system.retrieve_memories(
                        memory.content[:50], top_k=3
                    )
                    for sim_memory in similar_memories:
                        if (sim_memory.character_perspective not in memory.participants and
                            sim_memory.id != memory.id):
                            secret_leakage += 1
                            break
            
            online_metrics['secret_security'] = 1.0 - min(1.0, secret_leakage / max(1, len([m for m in recent_memories if m.is_secret])))
        
        return online_metrics


# Example usage and testing
if __name__ == "__main__":
    # This would typically be run after loading your memory_data.json
    # For demonstration, we'll create a minimal example
    
    from src.sekai_memory_system import SekaiMemorySystem
    from src.data_loader import MemoryDataProcessor
    
    # Initialize system
    memory_system = SekaiMemorySystem()
    
    # Load some sample data (you would use your actual memory_data.json)
    processor = MemoryDataProcessor(memory_system)
    
    # Initialize evaluation pipeline
    evaluator = ComprehensiveEvaluationPipeline(memory_system)
    
    # Run full evaluation
    results = evaluator.run_full_evaluation()
    
    # Generate report
    evaluator.generate_evaluation_report(results)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Overall Score: {results['overall_score']:.3f}")
    print(f"Retrieval Accuracy: {results['question_a_retrieval']['overall_score']:.3f}")
    print(f"Memory Consistency: {results['question_b_consistency']['overall_score']:.3f}")
    print(f"Additional Metrics: {results['question_c_additional']['overall_score']:.3f}")
    print("="*50)