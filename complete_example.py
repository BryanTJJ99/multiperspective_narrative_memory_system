# complete_example.py
#!/usr/bin/env python3
"""
Enhanced Complete Usage Example for Multiperspective Narrative Memory System
Now includes LLM integration for sophisticated character analysis and memory processing
"""

import json
import os
import asyncio
from datetime import datetime

# Import enhanced dynamic system as primary interface
from memory_system import MemorySystem, MemoryDataProcessor, ComprehensiveEvaluationPipeline
from env_config import env_config

def create_sample_data():
    """Create sample data for demonstration (subset of actual data)"""
    sample_data = [
        {
            "chapter_number": 1,
            "synopsis": "Byleth Eisner steps into Garreg Mach Corp, the air buzzing with first-day energy. Beneath a professional smile, a sharp mind is already at work, evaluating the office ecosystem. Key players like Dimitri, Sylvain, and Felix are noted - each a potential asset or obstacle in the games to come."
        },
        {
            "chapter_number": 2,
            "synopsis": "Amidst the cafeteria bustle, Byleth finds a quiet corner to observe. Focus locks onto Sylvain and Annette sharing an easy intimacy across the table. Their established relationship is clear - valuable intel. Byleth mentally files away this dynamic, already considering how it might be leveraged in future encounters."
        },
        {
            "chapter_number": 7,
            "synopsis": "Inside the privacy of Dimitri's office, the simmering tension boils over. Byleth makes the first move, initiating a kiss Dimitri returns with unexpected fervor. The professional boundary shatters, replaced by sudden, consuming passion. The affair ignites in the sterile quiet of the corporate building after dark."
        },
        {
            "chapter_number": 9,
            "synopsis": "Turning a corner near Dimitri's suite, Byleth nearly collides with Dedue, arriving early as always. His gaze sweeps over Byleth's slightly disheveled state, lingering for just a second too long. No words pass, but Byleth feels instantly, uncomfortably seen. A silent witness now exists."
        },
        {
            "chapter_number": 17,
            "synopsis": "Inside The Lost Saint's velvet shadows, Byleth meets Sylvain. Freed from the office's constraints, the air crackles with immediate, intense chemistry. Flirtation gives way to bold touches and charged conversation. The dynamic is wildly different from Dimitri - playful, adventurous, overtly physical from the start."
        },
        {
            "chapter_number": 26,
            "synopsis": "Later, Dedue is tidying Dimitri's living space as part of his duties. Near the sofa cushions, something catches his eye - a small, unique earring he recognizes as Byleth's. Concrete evidence confirming his earlier suspicion. His expression remains impassive, but the knowledge settles within him."
        },
        {
            "chapter_number": 31,
            "synopsis": "Sylvain pushes through the restaurant door, heading for takeout. His gaze sweeps the room and freezes, locking onto Dimitri's hand possessively covering Byleth's across the table. Shock, then thunder darkens his expression. He sees the betrayal playing out in public."
        },
        {
            "chapter_number": 38,
            "synopsis": "A company-wide health alert appears in everyone's inbox. It mentions monitoring a novel virus spreading rapidly overseas, advising standard hygiene practices. Just another corporate memo, easily lost in the daily flood. The significance is utterly lost on the preoccupied employees, including Byleth."
        }
    ]
    
    with open("sample_memory_data.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    
    return "sample_memory_data.json"

def get_openai_api_key():
    """Get OpenAI API key from environment configuration"""
    # Check if API key is already configured
    if env_config.openai_api_key:
        return env_config.openai_api_key
    
    print("\n" + "="*70)
    print("LLM INTEGRATION SETUP")
    print("="*70)
    print("No OpenAI API key found in environment variables or .env file.")
    print("To enable LLM features, you can:")
    print("1. Create a .env file and add OPENAI_API_KEY=your_key_here")
    print("2. Set the OPENAI_API_KEY environment variable")
    print("3. Continue without LLM features (statistical fallbacks will be used)")
    print("-"*70)
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice == "1":
        print("\nCreating .env file template...")
        env_config.create_sample_env_file()
        print("Please edit the .env file with your API key and restart the program.")
        return None
    elif choice == "2":
        print("Please set the OPENAI_API_KEY environment variable and restart.")
        return None
    else:
        print("Continuing without LLM features.")
        return None

async def demonstrate_enhanced_system_with_evaluation():
    """Comprehensive demonstration using Enhanced Dynamic Memory System with LLM integration"""
    
    print("🧠 Multiperspective Narrative Memory System")
    print("🚀 Enhanced Dynamic Memory System with LLM Integration")
    print("=" * 70)
    
    # Step 1: Initialize the enhanced dynamic system
    print("\n1️⃣  Initializing Enhanced Dynamic Memory System...")
    
    # Show current environment configuration
    env_config.print_configuration()
    
    # Validate configuration
    config_status = env_config.validate_configuration()
    if config_status['warnings']:
        print("\nConfiguration Warnings:")
        for warning in config_status['warnings']:
            print(f"   ⚠️  {warning}")
    
    memory_system = MemorySystem()  # Uses environment configuration automatically
    
    llm_status = "Enabled" if memory_system.llm_config.enabled else "Disabled"
    model_info = f" ({memory_system.llm_config.model})" if memory_system.llm_config.enabled else ""
    
    print(f"\n   ✅ Enhanced memory system initialized")
    print(f"   📊 Base characters: {len(memory_system.character_profiles)}")
    print(f"   🤖 LLM Integration: {llm_status}{model_info}")
    print(f"   📁 Auto-integration: Enabled")
    print(f"   👁 File watching: {'Enabled' if memory_system.watch_directory else 'Disabled'}")
    
    # Step 2: Load initial narrative data
    print("\n2️⃣  Loading Initial Narrative Data with LLM Enhancement...")
    
    # Check if user provided their actual data
    data_file = "memory_data.json" if os.path.exists("memory_data.json") else create_sample_data()
    
    if data_file == "memory_data.json":
        print("   📚 Using your actual memory_data.json file")
    else:
        print("   📚 Using sample data (add memory_data.json for your full story)")
    
    # Process data using enhanced system with environment configuration
    processor = MemoryDataProcessor(memory_system)  # No API key needed - uses env config
    stats = await processor.process_json_file(data_file)
    
    print(f"   ✅ Processed {stats['chapters_processed']} chapters")
    print(f"   ✅ Created {stats['memories_created']} memories")
    print(f"   ✅ Characters: {', '.join(stats['characters_involved'])}")
    print(f"   🤖 LLM Enhanced: {stats['llm_enhanced']}")
    
    # Step 3: Run comprehensive evaluation on enhanced system
    print("\n3️⃣  Running Comprehensive Evaluation...")
    
    evaluator = ComprehensiveEvaluationPipeline(memory_system)
    results = evaluator.run_full_evaluation()
    
    print(f"   🎯 Overall System Score: {results['overall_score']:.3f}")
    print(f"   📈 Retrieval Accuracy: {results['question_a_retrieval']['overall_score']:.3f}")
    print(f"   🔄 Memory Consistency: {results['question_b_consistency']['overall_score']:.3f}")
    print(f"   ⚡ Additional Metrics: {results['question_c_additional']['overall_score']:.3f}")
    
    # Step 4: Demonstrate enhanced dynamic capabilities
    print("\n4️⃣  Demonstrating Enhanced Dynamic Growth Capabilities...")
    
    # Add a new chapter with a new character
    # Fix: Handle empty narrative timeline
    existing_chapters = list(memory_system.narrative_timeline.keys())
    if existing_chapters:
        next_chapter_num = max(existing_chapters) + 1
    else:
        # Check memories for chapter numbers as fallback
        all_memories = list(memory_system.memories.values())
        if all_memories:
            next_chapter_num = max(m.chapter for m in all_memories) + 1
        else:
            next_chapter_num = 51  # Default starting point
    
    new_chapter = {
        "chapter_number": next_chapter_num,
        "synopsis": "Dr. Elena Vasquez arrives as the new head of security, immediately noticing inconsistencies in employee behavior patterns. Her analytical mind quickly identifies the complex web of relationships, particularly focusing on Byleth's interactions. 'Interesting dynamics here,' she notes, making mental notes of who avoids eye contact with whom."
    }
    
    print(f"   📝 Adding new chapter {new_chapter['chapter_number']} with new character...")
    integration_result = await memory_system.add_new_chapter(new_chapter, auto_integrate=True)
    
    print(f"   ✅ New memories created: {integration_result['new_memories_created']}")
    print(f"   👥 New characters discovered: {integration_result['new_characters_discovered']}")
    print(f"   🔄 Relationship changes: {len(integration_result['relationship_changes'])}")
    print(f"   ⚠️  Contradictions detected: {len(integration_result['contradictions_detected'])}")
    print(f"   🔧 Contradictions resolved: {len(integration_result['contradictions_resolved'])}")
    print(f"   🤖 LLM Enhanced: {integration_result['llm_enhanced']}")
    
    # Step 5: Re-evaluate system after growth
    print("\n5️⃣  Re-evaluating System After Dynamic Growth...")
    
    post_growth_results = evaluator.run_full_evaluation()
    
    print(f"   🎯 Updated Overall Score: {post_growth_results['overall_score']:.3f}")
    print(f"   📊 Score Change: {post_growth_results['overall_score'] - results['overall_score']:+.3f}")
    
    # Step 6: Enhanced character evolution analysis
    print("\n6️⃣  Enhanced Character Evolution Analysis...")
    
    growth_stats = memory_system.get_narrative_growth_stats()
    print(f"   📈 Total chapters: {growth_stats['total_chapters']}")
    print(f"   👥 Total characters: {growth_stats['total_characters']}")
    print(f"   🔗 Relationship network: {growth_stats['relationship_network_growth']['total_relationships']} connections")
    
    # LLM Enhancement Statistics
    llm_stats = growth_stats.get('llm_enhancement_stats', {})
    if llm_stats.get('llm_enabled'):
        print(f"   🤖 LLM Model: {llm_stats.get('llm_model_used', 'Unknown')}")
        print(f"   🧠 Characters analyzed: {llm_stats.get('characters_with_llm_analysis', 0)}")
        print(f"   📊 Average confidence: {llm_stats.get('average_analysis_confidence', 0):.3f}")
    
    # Analyze key characters with LLM insights
    key_characters = ["Byleth", "Dimitri", "Sylvain"]
    if "Elena" in memory_system.character_profiles:
        key_characters.append("Elena")
    
    print(f"   🎭 Enhanced Character Analysis:")
    for character in key_characters:
        if character in memory_system.character_profiles:
            evolution = memory_system.get_character_evolution(character)
            profile = memory_system.character_profiles[character]
            llm_analysis = evolution.get('llm_analysis', {})
            
            print(f"      {character}:")
            print(f"         Memories: {evolution['total_memories']}")
            print(f"         Relationships: {len(evolution['relationship_development'])}")
            
            if llm_analysis:
                traits = llm_analysis.get('personality_traits', [])
                emotional_state = llm_analysis.get('emotional_state', 'unknown')
                confidence = llm_analysis.get('analysis_confidence', 0)
                
                print(f"         LLM Traits: {', '.join(traits) if traits else 'Learning...'}")
                print(f"         Emotional State: {emotional_state}")
                print(f"         Analysis Confidence: {confidence:.2f}")
            else:
                print(f"         Traditional Traits: {', '.join(profile.personality_traits) if profile.personality_traits else 'Learning...'}")
            
            print(f"         Secrets: {len(profile.secret_knowledge)}")
    
    # Step 7: Demonstrate enhanced memory consolidation
    if memory_system.llm_config.enabled:
        print("\n7️⃣  Demonstrating LLM-Enhanced Memory Consolidation...")
        
        # Find some memories to consolidate
        byleth_memories = [m for m in memory_system.memories.values() if m.character_perspective == "Byleth"]
        if len(byleth_memories) >= 3:
            sample_group = byleth_memories[:3]
            print(f"   🔄 Consolidating {len(sample_group)} Byleth memories with LLM...")
            
            consolidated_id = await memory_system._consolidate_memory_group_enhanced(sample_group)
            if consolidated_id:
                consolidated_memory = memory_system.memories[consolidated_id]
                print(f"   ✅ Created consolidated memory: {consolidated_memory.content[:100]}...")
    
    # Step 8: Generate comprehensive report
    print("\n8️⃣  Generating Comprehensive Report...")
    
    report_filename = f"enhanced_narrative_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    evaluator.generate_evaluation_report(post_growth_results, report_filename)
    
    print(f"   📄 Report saved: {report_filename}")
    print(f"   🌐 Open in browser to view detailed analysis")
    
    # Step 9: Demonstrate enhanced memory queries
    print("\n9️⃣  Enhanced Memory Queries with LLM Insights...")
    
    sample_queries = [
        ("secret relationships and manipulation", "Byleth"),
        ("protective observations and evidence", "Dedue"),
        ("character introductions and first impressions", None),
        ("emotional conflicts and betrayal", None)
    ]
    
    for query, character_filter in sample_queries:
        print(f"\n   🔍 Query: '{query}'" + (f" (Character: {character_filter})" if character_filter else ""))
        
        memories = memory_system.retrieve_memories(
            query=query,
            character_perspective=character_filter,
            top_k=3
        )
        
        for i, memory in enumerate(memories, 1):
            emotion_icon = "😊" if memory.emotional_weight > 0 else "😐" if memory.emotional_weight == 0 else "😟"
            secret_icon = "🤫" if memory.is_secret else "👁️"
            
            print(f"      {i}. Chapter {memory.chapter} {emotion_icon} {secret_icon} ({memory.character_perspective})")
            print(f"         {memory.content[:80]}...")
            print(f"         Importance: {memory.importance:.2f} | Participants: {', '.join(memory.participants)}")
    
    # Step 10: Show enhanced system capabilities summary
    print("\n🔟  Enhanced System Capabilities Summary...")
    
    print(f"   🧠 Memory System Type: Dynamic (Auto-Growing) with LLM Enhancement")
    print(f"   📊 Total Memories Stored: {len(memory_system.memories)}")
    print(f"   👥 Character Profiles: {len(memory_system.character_profiles)}")
    print(f"   🔗 Interaction Graph Nodes: {memory_system.interaction_graph.number_of_nodes()}")
    print(f"   🔗 Interaction Graph Edges: {memory_system.interaction_graph.number_of_edges()}")
    print(f"   📈 Memory Updates Tracked: {len(memory_system.memory_updates)}")
    print(f"   🎯 Overall Evaluation Score: {post_growth_results['overall_score']:.3f}")
    
    if memory_system.llm_config.enabled:
        print(f"   🤖 LLM Model: {memory_system.llm_config.model}")
        print(f"   🧠 Characters with LLM Analysis: {llm_stats.get('characters_with_llm_analysis', 0)}")
        print(f"   📊 Average LLM Confidence: {llm_stats.get('average_analysis_confidence', 0):.3f}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("✨ ENHANCED DYNAMIC MEMORY SYSTEM DEMONSTRATION COMPLETE")
    print("=" * 70)
    
    print(f"🎯 System Performance:")
    print(f"   • Overall Score: {post_growth_results['overall_score']:.3f}")
    print(f"   • Retrieval Accuracy: {post_growth_results['question_a_retrieval']['overall_score']:.3f}")
    print(f"   • Memory Consistency: {post_growth_results['question_b_consistency']['overall_score']:.3f}")
    print(f"   • Advanced Metrics: {post_growth_results['question_c_additional']['overall_score']:.3f}")
    
    print(f"\n📊 Narrative Statistics:")
    print(f"   • Chapters: {growth_stats['total_chapters']}")
    print(f"   • Characters: {growth_stats['total_characters']}")
    print(f"   • Memories: {len(memory_system.memories)}")
    print(f"   • Relationships: {growth_stats['relationship_network_growth']['total_relationships']}")
    print(f"   • Secrets: {sum(1 for m in memory_system.memories.values() if m.is_secret)}")
    
    if memory_system.llm_config.enabled:
        print(f"\n🤖 LLM Enhancement:")
        print(f"   • Model Used: {memory_system.llm_config.model}")
        print(f"   • Characters Analyzed: {llm_stats.get('characters_with_llm_analysis', 0)}")
        print(f"   • Average Confidence: {llm_stats.get('average_analysis_confidence', 0):.3f}")
    
    print(f"\n📁 Files Generated:")
    print(f"   • Database: enhanced_narrative_memory.db")
    print(f"   • Evaluation Report: {report_filename}")
    if data_file == "sample_memory_data.json":
        print(f"   • Sample Data: sample_memory_data.json")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Add your full memory_data.json for complete analysis")
    print(f"   2. Use add_new_chapter() to grow your narrative dynamically")
    print(f"   3. Enable file watching for automatic updates")
    print(f"   4. Integrate with your application via enhanced API")
    print(f"   5. Monitor character evolution with LLM insights")
    if not memory_system.llm_config.enabled:
        print(f"   6. Add OpenAI API key to enable LLM features")
    
    return memory_system, post_growth_results

async def interactive_demo():
    """Interactive demonstration allowing user queries with LLM enhancement"""
    
    print("\n" + "=" * 70)
    print("🎮 INTERACTIVE DEMO MODE WITH LLM ENHANCEMENT")
    print("=" * 70)
    print("Type your queries to search the enhanced memory system!")
    print("Commands:")
    print("  • 'quit' or 'exit' to stop")
    print("  • 'stats' for system statistics")
    print("  • 'characters' to list all characters with LLM insights")
    print("  • 'secrets' to show secret memories")
    print("  • 'llm' to show LLM enhancement status")
    print("  • 'add [synopsis]' to add a new chapter")
    print("  • Any other text will search memories")
    print("-" * 70)
    
    memory_system, _ = await demonstrate_enhanced_system_with_evaluation()
    
    while True:
        query = input("\n🔍 Enter query: ").strip()
        
        if query.lower() in ['quit', 'exit']:
            print("👋 Goodbye!")
            break
        
        elif query.lower() == 'stats':
            stats = memory_system.get_narrative_growth_stats()
            print(f"📊 System Statistics:")
            print(f"   • Total chapters: {stats['total_chapters']}")
            print(f"   • Total memories: {stats['total_memories']}")
            print(f"   • Characters: {stats['total_characters']}")
            print(f"   • Relationships: {stats['relationship_network_growth']['total_relationships']}")
            print(f"   • Recent updates: {stats['recent_updates']}")
            
            llm_stats = stats.get('llm_enhancement_stats', {})
            if llm_stats.get('llm_enabled'):
                print(f"   • LLM Model: {llm_stats.get('llm_model_used', 'Unknown')}")
                print(f"   • LLM Analyzed Characters: {llm_stats.get('characters_with_llm_analysis', 0)}")
        
        elif query.lower() == 'characters':
            print("👥 Characters in system with LLM insights:")
            for character in memory_system.character_profiles:
                evolution = memory_system.get_character_evolution(character)
                profile = memory_system.character_profiles[character]
                llm_analysis = evolution.get('llm_analysis', {})
                
                print(f"   • {character}: {evolution['total_memories']} memories, {len(profile.secret_knowledge)} secrets")
                
                if llm_analysis and llm_analysis.get('analysis_confidence', 0) > 0:
                    traits = llm_analysis.get('personality_traits', [])
                    emotional_state = llm_analysis.get('emotional_state', 'unknown')
                    confidence = llm_analysis.get('analysis_confidence', 0)
                    print(f"     LLM Analysis: {', '.join(traits)} (Confidence: {confidence:.2f})")
                    print(f"     Emotional State: {emotional_state}")
                else:
                    print(f"     Traditional Traits: {', '.join(profile.personality_traits) if profile.personality_traits else 'Learning...'}")
        
        elif query.lower() == 'secrets':
            secret_memories = [m for m in memory_system.memories.values() if m.is_secret]
            print(f"🤫 Secret memories ({len(secret_memories)} total):")
            for memory in secret_memories[:5]:  # Show first 5
                print(f"   • Chapter {memory.chapter} ({memory.character_perspective}): {memory.content[:80]}...")
        
        elif query.lower() == 'llm':
            print(f"🤖 LLM Enhancement Status:")
            print(f"   • Enabled: {memory_system.llm_config.enabled}")
            if memory_system.llm_config.enabled:
                print(f"   • Model: {memory_system.llm_config.model}")
                print(f"   • Temperature: {memory_system.llm_config.temperature}")
                stats = memory_system.get_narrative_growth_stats()
                llm_stats = stats.get('llm_enhancement_stats', {})
                print(f"   • Characters Analyzed: {llm_stats.get('characters_with_llm_analysis', 0)}")
                print(f"   • Average Confidence: {llm_stats.get('average_analysis_confidence', 0):.3f}")
            else:
                print(f"   • Reason: No OpenAI API key provided")
                print(f"   • Fallback: Using statistical analysis")
        
        elif query.lower().startswith('add '):
            synopsis = query[4:].strip()
            if synopsis:
                # Get next chapter number
                existing_chapters = list(memory_system.narrative_timeline.keys())
                next_chapter = max(existing_chapters) + 1 if existing_chapters else 1
                
                new_chapter = {
                    "chapter_number": next_chapter,
                    "synopsis": synopsis
                }
                
                print(f"📝 Adding chapter {next_chapter}...")
                result = await memory_system.add_new_chapter(new_chapter)
                print(f"✅ Added {result['new_memories_created']} memories")
                if result['new_characters_discovered']:
                    print(f"👥 New characters: {', '.join(result['new_characters_discovered'])}")
                print(f"🤖 LLM Enhanced: {result.get('llm_enhanced', False)}")
            else:
                print("⚠️ Please provide a synopsis after 'add'")
        
        elif query:
            memories = memory_system.retrieve_memories(query, top_k=3)
            if memories:
                print(f"🎯 Found {len(memories)} relevant memories:")
                for i, memory in enumerate(memories, 1):
                    emotion = "😊" if memory.emotional_weight > 0 else "😐" if memory.emotional_weight == 0 else "😟"
                    secret = "🤫" if memory.is_secret else "👁️"
                    print(f"   {i}. Chapter {memory.chapter} {emotion} {secret} ({memory.character_perspective})")
                    print(f"      {memory.content[:100]}...")
                    print(f"      Importance: {memory.importance:.2f} | Participants: {', '.join(memory.participants)}")
            else:
                print("⚠️ No memories found for that query. Try different keywords!")

async def main():
    """Main demonstration function with async support"""
    
    print("🧠 Multiperspective Narrative Memory System")
    print("🎯 Enhanced Dynamic Memory System with LLM Integration")
    print("=" * 70)
    
    mode = input("""
Choose demonstration mode:
1. Full Enhanced System with Evaluation (Recommended)
2. Interactive Query Mode with LLM Features
3. Both Demonstrations

Enter choice (1-3): """).strip()
    
    if mode == "1":
        memory_system, results = await demonstrate_enhanced_system_with_evaluation()
        print(f"\n🎉 Enhanced dynamic system ready for production use!")
        
    elif mode == "2":
        await interactive_demo()
        print(f"\n🎮 Interactive demo completed")
        
    elif mode == "3":
        print("\n🚀 Running full demonstration then interactive mode...")
        memory_system, results = await demonstrate_enhanced_system_with_evaluation()
        
        continue_interactive = input("\n🎮 Continue with interactive mode? (y/n): ").strip().lower()
        if continue_interactive == 'y':
            await interactive_demo()
        
    else:
        print("⚠️ Invalid choice. Running full demonstration...")
        memory_system, results = await demonstrate_enhanced_system_with_evaluation()
    
    print("\n" + "=" * 70)
    print("🎉 Thank you for exploring the Enhanced Dynamic Memory System!")
    print("📚 The system is now ready for your narrative with LLM capabilities!")
    print("=" * 70)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n⚠️ Error during demonstration: {e}")
        print("📞 Please check the system requirements and API key, then try again.")
        raise  # Re-raise for debugging