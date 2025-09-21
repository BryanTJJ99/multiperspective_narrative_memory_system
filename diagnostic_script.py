# diagnostic_script.py
"""
Diagnostic script to identify why LLM character analysis isn't running
"""

import asyncio
import json
from memory_system import MemorySystem, MemoryDataProcessor

async def diagnose_character_analysis():
    """Diagnose LLM character analysis issues"""
    
    print("🔍 LLM Character Analysis Diagnostic")
    print("=" * 50)
    
    # Initialize system
    memory_system = MemorySystem()
    
    # Check LLM configuration
    print(f"1. LLM Configuration:")
    print(f"   ✓ LLM Enabled: {memory_system.llm_config.enabled}")
    print(f"   ✓ Model: {memory_system.llm_config.model}")
    print(f"   ✓ API Key Present: {'Yes' if memory_system.llm_config.api_key else 'No'}")
    
    # Test LLM connection
    print(f"\n2. Testing LLM Connection:")
    try:
        connection_ok = await memory_system._test_llm_connection()
        print(f"   ✓ Connection Test: {'PASSED' if connection_ok else 'FAILED'}")
        
        if not connection_ok:
            print(f"   ✗ Last Error: {memory_system.llm_stats.get('last_error', 'Unknown')}")
    except Exception as e:
        print(f"   ✗ Connection Error: {e}")
    
    # Check character profiles
    print(f"\n3. Character Profile Status:")
    for char_name, profile in memory_system.character_profiles.items():
        char_memories = [m for m in memory_system.memories.values() if m.character_perspective == char_name]
        print(f"   {char_name}: {len(char_memories)} memories, confidence: {profile.llm_analysis_confidence:.2f}")
    
    # Get LLM diagnostics
    print(f"\n4. LLM Usage Statistics:")
    diagnostics = memory_system.get_llm_diagnostics()
    for key, value in diagnostics.items():
        print(f"   {key}: {value}")
    
    # Test manual character analysis
    print(f"\n5. Testing Manual Character Analysis:")
    byleth_memories = [m for m in memory_system.memories.values() if m.character_perspective == "Byleth"]
    
    if byleth_memories:
        print(f"   Found {len(byleth_memories)} Byleth memories")
        print(f"   Testing LLM analysis on Byleth...")
        
        try:
            analysis_result = await memory_system._llm_analyze_character("Byleth", byleth_memories[:5])
            print(f"   ✓ Analysis Result: {json.dumps(analysis_result, indent=2)}")
            
        except Exception as e:
            print(f"   ✗ Analysis Failed: {e}")
    
    # Check when character analysis should be triggered
    print(f"\n6. Character Analysis Trigger Points:")
    print(f"   During data loading: Should trigger after processing memories")
    print(f"   During new chapter addition: Should trigger if significant new info")
    print(f"   During character profile updates: Should trigger for re-analysis")
    
    # Look for the actual trigger calls
    print(f"\n7. Checking Integration Points:")
    
    # Test the update character profiles enhanced method
    try:
        print(f"   Testing character profile update...")
        mentioned_characters = {"Byleth", "Dimitri"}
        new_memory_ids = list(memory_system.memories.keys())[-3:]  # Last 3 memories
        
        await memory_system._update_character_profiles_enhanced(new_memory_ids, mentioned_characters)
        print(f"   ✓ Character profile update completed")
        
        # Check if analysis was performed
        updated_diagnostics = memory_system.get_llm_diagnostics()
        print(f"   Characters analyzed after update: {updated_diagnostics['characters_analyzed']}")
        
    except Exception as e:
        print(f"   ✗ Character profile update failed: {e}")

if __name__ == "__main__":
    asyncio.run(diagnose_character_analysis())