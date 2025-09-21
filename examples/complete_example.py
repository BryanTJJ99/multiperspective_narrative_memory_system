#!/usr/bin/env python3
"""
Complete Usage Example for Sekai Memory System
Demonstrates full workflow from data loading to evaluation
"""

import json
import os

# Ensure the repository root is on sys.path so `from src...` imports work when running the script
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation_pipeline import ComprehensiveEvaluationPipeline
from src.data_loader import MemoryDataProcessor

# Import all necessary components
from src.sekai_memory_system import MemoryType, RelationshipType, SekaiMemorySystem


def create_sample_data():
    """Create sample data for demonstration (subset of your actual data)"""
    # Use the repository's canonical data file instead of writing a new sample
    data_path = os.path.join(ROOT, "data", "memory_data.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Expected data file not found: {data_path}")
    return data_path


def demonstrate_memory_system():
    """Complete demonstration of the Sekai Memory System"""

    print("🧠 Sekai Memory System - Complete Demonstration")
    print("=" * 60)

    # Step 1: Initialize the system
    print("\n1️⃣  Initializing Memory System...")
    memory_system = SekaiMemorySystem(db_path="demo_sekai_memory.db")
    print(
        f"   ✅ Memory system initialized with {len(memory_system.core_memory)} characters"
    )

    # Step 2: Create and load sample data
    print("\n2️⃣  Loading Chapter Data...")
    data_file = create_sample_data()
    processor = MemoryDataProcessor(memory_system)
    stats = processor.process_json_file(data_file)

    print(f"   ✅ Processed {stats['chapters_processed']} chapters")
    print(f"   ✅ Created {stats['memories_created']} memories")
    print(f"   ✅ Characters involved: {', '.join(stats['characters_involved'])}")

    # Step 3: Demonstrate memory retrieval
    print("\n3️⃣  Demonstrating Memory Retrieval...")

    # Query 1: Byleth's relationship with Dimitri
    print("\n   🔍 Searching: 'Byleth Dimitri relationship affair'")
    byleth_dimitri_memories = memory_system.retrieve_memories(
        "Byleth Dimitri relationship affair", character_perspective="Byleth", top_k=3
    )

    for i, memory in enumerate(byleth_dimitri_memories, 1):
        print(
            f"      {i}. Chapter {memory.chapter} | Importance: {memory.importance:.2f} | Secret: {memory.is_secret}"
        )
        print(f"         {memory.content[:100]}...")

    # Query 2: Dedue's observations
    print("\n   🔍 Searching: 'Dedue observation evidence witness'")
    dedue_memories = memory_system.retrieve_memories(
        "Dedue observation evidence witness", character_perspective="Dedue", top_k=3
    )

    for i, memory in enumerate(dedue_memories, 1):
        print(
            f"      {i}. Chapter {memory.chapter} | Emotional Weight: {memory.emotional_weight:.2f}"
        )
        print(f"         {memory.content[:100]}...")

    # Step 4: Demonstrate relationship analysis
    print("\n4️⃣  Analyzing Character Relationships...")

    # Byleth-Dimitri relationship timeline
    timeline = memory_system.get_character_relationship_timeline("Byleth", "Dimitri")
    print(
        f"\n   📈 Byleth-Dimitri Relationship Timeline ({len(timeline)} interactions):"
    )

    for memory in timeline:
        emotion_indicator = (
            "😊"
            if memory.emotional_weight > 0
            else "😐" if memory.emotional_weight == 0 else "😟"
        )
        secret_indicator = "🤫" if memory.is_secret else "👁️"
        print(
            f"      Chapter {memory.chapter:2d} {emotion_indicator} {secret_indicator} | Weight: {memory.emotional_weight:+.2f}"
        )

    # Step 5: Demonstrate consistency checking
    print("\n5️⃣  Checking Memory Consistency...")

    consistency_report = memory_system.analyze_memory_consistency()
    print(f"   📊 Total memories analyzed: {consistency_report['total_memories']}")
    print(f"   ⚠️  Contradictions found: {len(consistency_report['contradictions'])}")

    if consistency_report["contradictions"]:
        print("   🔍 Contradiction details:")
        for memory_id, contradicting_ids in list(
            consistency_report["contradictions"].items()
        )[:2]:
            memory = memory_system.memories[memory_id]
            print(
                f"      Memory {memory_id[:8]}... (Chapter {memory.chapter}) conflicts with {len(contradicting_ids)} others"
            )

    # Character consistency analysis
    if consistency_report["character_consistency"]:
        print(f"   👥 Character consistency analysis:")
        for character, consistency_data in consistency_report[
            "character_consistency"
        ].items():
            sentiment_drift = consistency_data["sentiment_drift"]
            consistency_level = (
                "Good"
                if sentiment_drift < 0.5
                else "Moderate" if sentiment_drift < 1.0 else "Poor"
            )
            print(
                f"      {character}: {consistency_level} (drift: {sentiment_drift:.3f})"
            )

    # Step 6: Demonstrate secret management
    print("\n6️⃣  Analyzing Secret Management...")

    all_memories = list(memory_system.memories.values())
    secret_memories = [m for m in all_memories if m.is_secret]

    print(f"   🤫 Total secrets: {len(secret_memories)}")

    # Analyze secrets by character
    secret_holders = {}
    for memory in secret_memories:
        perspective = memory.character_perspective
        if perspective not in secret_holders:
            secret_holders[perspective] = 0
        secret_holders[perspective] += 1

    print("   📊 Secrets by character:")
    for character, count in sorted(
        secret_holders.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"      {character}: {count} secrets")

    # Step 7: Demonstrate graph analysis
    print("\n7️⃣  Character Interaction Graph Analysis...")

    import networkx as nx

    # Character centrality in interaction graph
    if memory_system.interaction_graph.nodes():
        centrality = nx.degree_centrality(memory_system.interaction_graph)
        print("   🌐 Character centrality (connectedness):")
        for character, score in sorted(
            centrality.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"      {character}: {score:.3f}")

    # Step 8: Run comprehensive evaluation
    print("\n8️⃣  Running Comprehensive Evaluation...")

    evaluator = ComprehensiveEvaluationPipeline(memory_system)
    results = evaluator.run_full_evaluation()

    print(f"   🎯 Overall Score: {results['overall_score']:.3f}")
    print(
        f"   📈 Retrieval Accuracy: {results['question_a_retrieval']['overall_score']:.3f}"
    )
    print(
        f"      - Precision: {results['question_a_retrieval']['average_precision']:.3f}"
    )
    print(f"      - Recall: {results['question_a_retrieval']['average_recall']:.3f}")
    print(f"      - F1 Score: {results['question_a_retrieval']['average_f1']:.3f}")

    print(
        f"   🔄 Memory Consistency: {results['question_b_consistency']['overall_score']:.3f}"
    )
    print(
        f"      - Temporal: {results['question_b_consistency']['temporal_consistency']['consistency_score']:.3f}"
    )
    print(
        f"      - Cross-Character: {results['question_b_consistency']['cross_character_consistency']['consistency_score']:.3f}"
    )
    print(
        f"      - World State: {results['question_b_consistency']['world_state_consistency']['consistency_score']:.3f}"
    )

    print(
        f"   ⚡ Additional Metrics: {results['question_c_additional']['overall_score']:.3f}"
    )
    print(
        f"      - Character Distinctiveness: {results['question_c_additional']['character_distinctiveness']['distinctiveness_score']:.3f}"
    )
    print(
        f"      - Secret Management: {results['question_c_additional']['secret_management']['secret_containment_score']:.3f}"
    )
    print(
        f"      - Relationship Tracking: {results['question_c_additional']['relationship_tracking']['avg_relationship_score']:.3f}"
    )

    # Step 9: Generate detailed report
    print("\n9️⃣  Generating Evaluation Report...")

    report_filename = (
        f"sekai_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    evaluator.generate_evaluation_report(results, report_filename)
    print(f"   📄 Detailed report saved to: {report_filename}")

    # Step 10: Demonstrate advanced queries
    print("\n🔟 Advanced Query Examples...")

    # Multi-character perspective on same event
    print("\n   🎭 Multi-perspective analysis of Chapter 31 (restaurant scene):")
    restaurant_memories = memory_system.retrieve_memories(
        "restaurant Dimitri hand table", chapter_range=(30, 32), top_k=5
    )

    perspectives = {}
    for memory in restaurant_memories:
        char = memory.character_perspective
        if char not in perspectives:
            perspectives[char] = []
        perspectives[char].append(memory)

    for character, memories in perspectives.items():
        if memories:
            memory = memories[0]  # Take the most relevant
            emotion = (
                "😊"
                if memory.emotional_weight > 0
                else "😐" if memory.emotional_weight == 0 else "😟"
            )
            print(f"      {character} {emotion}: {memory.content[:80]}...")

    # Temporal query - early vs late story
    print("\n   ⏰ Comparing early vs late story memories:")

    early_memories = [m for m in all_memories if m.chapter <= 10]
    late_memories = [m for m in all_memories if m.chapter >= 30]

    early_sentiment = (
        sum(m.emotional_weight for m in early_memories) / len(early_memories)
        if early_memories
        else 0
    )
    late_sentiment = (
        sum(m.emotional_weight for m in late_memories) / len(late_memories)
        if late_memories
        else 0
    )

    print(f"      Early story sentiment: {early_sentiment:+.3f}")
    print(f"      Late story sentiment: {late_sentiment:+.3f}")
    print(f"      Sentiment shift: {late_sentiment - early_sentiment:+.3f}")

    # World state evolution
    print("\n   🌍 World state evolution:")
    world_memories = [
        m for m in all_memories if m.relationship_type == RelationshipType.WORLD_MEMORY
    ]
    world_memories.sort(key=lambda x: x.chapter)

    for memory in world_memories:
        print(f"      Chapter {memory.chapter}: {memory.content[:80]}...")

    # Final summary
    print("\n" + "=" * 60)
    print("✨ DEMONSTRATION COMPLETE")
    print("=" * 60)
    print(f"📊 System Statistics:")
    print(f"   • Total memories: {len(memory_system.memories)}")
    print(f"   • Characters tracked: {len(memory_system.core_memory)}")
    print(f"   • Secret memories: {len(secret_memories)}")
    chapter_nums = sorted({m.chapter for m in all_memories})
    print(f"   • Chapter numbers: {', '.join(str(c) for c in chapter_nums)}")
    print(f"   • Overall system score: {results['overall_score']:.3f}")

    print(f"\n📁 Files created:")
    print(f"   • Database: demo_sekai_memory.db")
    print(f"   • Sample data: sample_memory_data.json")
    print(f"   • Evaluation report: {report_filename}")

    print(f"\n🎯 Key Features Demonstrated:")
    print(f"   ✅ Multi-character memory perspectives")
    print(f"   ✅ Secret compartmentalization")
    print(f"   ✅ Relationship timeline tracking")
    print(f"   ✅ Memory consistency validation")
    print(f"   ✅ Comprehensive evaluation pipeline")
    print(f"   ✅ Graph-based character analysis")

    return memory_system, results


def interactive_demo():
    """Interactive demonstration allowing user queries"""

    print("\n" + "=" * 60)
    print("🎮 INTERACTIVE DEMO MODE")
    print("=" * 60)
    print("Type your queries to search the memory system!")
    print("Commands:")
    print("  • 'quit' or 'exit' to stop")
    print("  • 'stats' for system statistics")
    print("  • 'characters' to list all characters")
    print("  • 'secrets' to show secret memories")
    print("  • Any other text will search memories")
    print("-" * 60)

    memory_system, _ = demonstrate_memory_system()

    while True:
        query = input("\n🔍 Enter query: ").strip()

        if query.lower() in ["quit", "exit"]:
            print("👋 Goodbye!")
            break

        elif query.lower() == "stats":
            all_memories = list(memory_system.memories.values())
            print(f"📊 System Statistics:")
            print(f"   • Total memories: {len(all_memories)}")
            print(f"   • Characters: {', '.join(memory_system.core_memory.keys())}")
            print(f"   • Secrets: {len([m for m in all_memories if m.is_secret])}")
            chapter_nums = sorted({m.chapter for m in all_memories})
            print(f"   • Chapter numbers: {', '.join(str(c) for c in chapter_nums)}")

        elif query.lower() == "characters":
            print("👥 Characters in system:")
            for character in memory_system.core_memory.keys():
                char_memories = [
                    m
                    for m in memory_system.memories.values()
                    if m.character_perspective == character
                ]
                secrets = len([m for m in char_memories if m.is_secret])
                print(
                    f"   • {character}: {len(char_memories)} memories ({secrets} secrets)"
                )

        elif query.lower() == "secrets":
            secret_memories = [
                m for m in memory_system.memories.values() if m.is_secret
            ]
            print(f"🤫 Secret memories ({len(secret_memories)} total):")
            for memory in secret_memories[:5]:  # Show first 5
                print(
                    f"   • Chapter {memory.chapter} | {memory.character_perspective}: {memory.content[:80]}..."
                )

        if __name__ == "__main__":
            # Default: run the full demonstration
            demonstrate_memory_system()