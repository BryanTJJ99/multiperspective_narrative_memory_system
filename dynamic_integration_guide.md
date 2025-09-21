<!-- dynamic_integration_guide.md -->
# Dynamic Memory System Integration Guide

## 🚀 Advanced Integration for Evolving Narratives

The **DynamicMemorySystem** provides sophisticated capabilities for narratives that grow and evolve over time. This guide covers advanced integration patterns, production deployment, and optimization strategies.

## 🎯 Core Dynamic Capabilities

### 1. **Automatic Chapter Integration**

```python
from memory_system import MemorySystem

# Initialize with dynamic capabilities
memory_system = MemorySystem(
    db_path="production_narrative.db",
    watch_directory="./story_chapters"  # Optional: auto-processing
)

# Add chapters programmatically
new_chapter = {
    "chapter_number": 51,
    "synopsis": "Dr. Elena Vasquez arrives as the new head of security, immediately noticing inconsistencies in employee behavior patterns that suggest hidden relationships."
}

result = memory_system.add_new_chapter(new_chapter, auto_integrate=True)

print(f"✅ Integration Results:")
print(f"   Memories created: {result['new_memories_created']}")
print(f"   New characters: {result['new_characters_discovered']}")
print(f"   Relationship changes: {len(result['relationship_changes'])}")
print(f"   Contradictions resolved: {len(result['contradictions_resolved'])}")
```

### 2. **Character Evolution Tracking**

```python
# Monitor how characters develop over time
def analyze_character_growth(character_name):
    evolution = memory_system.get_character_evolution(character_name)
    
    return {
        "character": character_name,
        "introduction_chapter": evolution['introduction_chapter'],
        "total_interactions": evolution['total_memories'],
        "relationship_count": len(evolution['relationship_development']),
        "emotional_arc": evolution['emotional_journey'],
        "personality_development": evolution['personality_evolution'],
        "secret_involvement": len(evolution['secret_knowledge_progression'])
    }

# Example: Track Byleth's development
byleth_growth = analyze_character_growth("Byleth")
print(f"Byleth has {byleth_growth['relationship_count']} relationships")
print(f"Emotional journey span: {len(byleth_growth['emotional_arc'])} chapters")
```

### 3. **Real-Time File Monitoring**

```python
import time
from pathlib import Path

# Set up automatic narrative processing
def setup_automatic_processing():
    memory_system = MemorySystem(
        db_path="live_narrative.db",
        watch_directory="./live_chapters"
    )
    
    # Start file watcher
    memory_system.start_file_watcher(check_interval=30)  # Check every 30 seconds
    
    print("📡 File watcher active - drop new chapter files to auto-process!")
    return memory_system

# Example: Create new chapter file
def add_chapter_file(chapter_num, synopsis):
    chapter_data = {
        "chapter_number": chapter_num,
        "synopsis": synopsis
    }
    
    file_path = Path("./live_chapters") / f"chapter_{chapter_num}.json"
    file_path.parent.mkdir(exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(chapter_data, f, indent=2)
    
    print(f"📝 Created {file_path} - will be auto-processed!")

# Usage
memory_system = setup_automatic_processing()
add_chapter_file(52, "Elena's first day reveals suspicious scheduling patterns...")
```

## 🔧 Production Integration Patterns

### Pattern 1: **Web Application Integration**

```python
from flask import Flask, request, jsonify
from memory_system import MemorySystem, create_flask_app

class NarrativeWebApp:
    def __init__(self):
        self.memory_system = MemorySystem(db_path="webapp_narrative.db")
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route('/story/add_chapter', methods=['POST'])
        def add_chapter():
            chapter_data = request.json
            
            # Validate chapter data
            if not self._validate_chapter(chapter_data):
                return jsonify({"error": "Invalid chapter format"}), 400
            
            # Process chapter
            result = self.memory_system.add_new_chapter(chapter_data)
            
            # Return integration results
            return jsonify({
                "success": True,
                "chapter_number": result['chapter_number'],
                "memories_created": result['new_memories_created'],
                "new_characters": result['new_characters_discovered'],
                "processing_time": result['processing_time']
            })
        
        @self.app.route('/story/characters/<character_name>/evolution', methods=['GET'])
        def get_character_evolution(character_name):
            evolution = self.memory_system.get_character_evolution(character_name)
            if "error" in evolution:
                return jsonify(evolution), 404
            return jsonify(evolution)
        
        @self.app.route('/story/search', methods=['POST'])
        def search_narrative():
            data = request.json
            query = data.get('query', '')
            filters = data.get('filters', {})
            
            memories = self.memory_system.retrieve_memories(
                query=query,
                character_perspective=filters.get('character'),
                chapter_range=filters.get('chapter_range'),
                top_k=filters.get('limit', 10)
            )
            
            return jsonify({
                "query": query,
                "results": [self._format_memory(m) for m in memories],
                "count": len(memories)
            })
    
    def _validate_chapter(self, chapter_data):
        return (isinstance(chapter_data, dict) and 
                'chapter_number' in chapter_data and 
                'synopsis' in chapter_data)
    
    def _format_memory(self, memory):
        return {
            "id": memory.id,
            "character": memory.character_perspective,
            "content": memory.content,
            "chapter": memory.chapter,
            "participants": memory.participants,
            "emotional_weight": memory.emotional_weight,
            "importance": memory.importance,
            "is_secret": memory.is_secret
        }

# Deploy web app
app = NarrativeWebApp()
if __name__ == "__main__":
    app.app.run(host='0.0.0.0', port=5000, debug=False)
```

### Pattern 2: **Background Processing Service**

```python
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from memory_system import MemorySystem

class NarrativeProcessingService:
    def __init__(self, config):
        self.memory_system = MemorySystem(
            db_path=config['db_path'],
            watch_directory=config['watch_directory']
        )
        self.executor = ThreadPoolExecutor(max_workers=config.get('workers', 4))
        self.processing_queue = asyncio.Queue()
        self.running = False
    
    async def start(self):
        """Start the background processing service"""
        self.running = True
        
        # Start file watcher
        self.memory_system.start_file_watcher(check_interval=30)
        
        # Start processing loop
        await asyncio.gather(
            self.process_queue(),
            self.monitor_health(),
            self.periodic_consolidation()
        )
    
    async def process_queue(self):
        """Process chapters from the queue"""
        while self.running:
            try:
                chapter_data = await asyncio.wait_for(
                    self.processing_queue.get(), timeout=1.0
                )
                
                # Process in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    self.memory_system.add_new_chapter,
                    chapter_data
                )
                
                logging.info(f"Processed chapter {result['chapter_number']}: "
                           f"{result['new_memories_created']} memories created")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Error processing chapter: {e}")
    
    async def monitor_health(self):
        """Monitor system health and performance"""
        while self.running:
            try:
                stats = self.memory_system.get_narrative_growth_stats()
                
                # Log health metrics
                logging.info(f"System health: {stats['total_memories']} memories, "
                           f"{stats['total_characters']} characters")
                
                # Alert on issues
                if stats.get('recent_updates', 0) == 0:
                    logging.warning("No recent updates - check file watcher")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logging.error(f"Health check failed: {e}")
                await asyncio.sleep(60)
    
    async def periodic_consolidation(self):
        """Perform periodic memory consolidation"""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                # Check if consolidation is needed
                memory_count = len(self.memory_system.memories)
                if memory_count > 1000:  # Consolidate large memory stores
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        self.executor,
                        self._perform_consolidation
                    )
                    logging.info("Performed memory consolidation")
                    
            except Exception as e:
                logging.error(f"Consolidation error: {e}")
    
    def _perform_consolidation(self):
        """Perform memory consolidation"""
        # Get recent chapters for consolidation
        recent_chapters = sorted(self.memory_system.narrative_timeline.keys())[-10:]
        for chapter in recent_chapters:
            self.memory_system._trigger_memory_consolidation(chapter)
    
    async def add_chapter(self, chapter_data):
        """Add chapter to processing queue"""
        await self.processing_queue.put(chapter_data)
    
    def stop(self):
        """Stop the service"""
        self.running = False
        self.memory_system.stop_file_watcher()
        self.executor.shutdown(wait=True)

# Usage
async def main():
    config = {
        'db_path': 'production_narrative.db',
        'watch_directory': './incoming_chapters',
        'workers': 4
    }
    
    service = NarrativeProcessingService(config)
    await service.start()

# Run service
# asyncio.run(main())
```

### Pattern 3: **Batch Processing Integration**

```python
from memory_system import MemorySystem
import json
from pathlib import Path
import logging

class BatchNarrativeProcessor:
    def __init__(self, config):
        self.memory_system = MemorySystem(db_path=config['db_path'])
        self.batch_size = config.get('batch_size', 10)
        self.results = []
    
    def process_chapter_directory(self, directory_path):
        """Process all chapters in a directory"""
        chapter_files = sorted(Path(directory_path).glob("*.json"))
        
        print(f"📁 Processing {len(chapter_files)} chapter files...")
        
        for i in range(0, len(chapter_files), self.batch_size):
            batch = chapter_files[i:i + self.batch_size]
            self._process_batch(batch)
            
            print(f"✅ Processed batch {i // self.batch_size + 1}/{(len(chapter_files) + self.batch_size - 1) // self.batch_size}")
        
        return self._generate_batch_report()
    
    def _process_batch(self, file_batch):
        """Process a batch of chapter files"""
        batch_results = []
        
        for file_path in file_batch:
            try:
                with open(file_path, 'r') as f:
                    chapter_data = json.load(f)
                
                # Handle both single chapters and arrays
                chapters = chapter_data if isinstance(chapter_data, list) else [chapter_data]
                
                for chapter in chapters:
                    result = self.memory_system.add_new_chapter(chapter)
                    batch_results.append({
                        'file': file_path.name,
                        'chapter': result['chapter_number'],
                        'memories': result['new_memories_created'],
                        'new_characters': result['new_characters_discovered'],
                        'processing_time': result['processing_time']
                    })
                    
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
                batch_results.append({
                    'file': file_path.name,
                    'error': str(e)
                })
        
        self.results.extend(batch_results)
    
    def _generate_batch_report(self):
        """Generate processing report"""
        successful = [r for r in self.results if 'error' not in r]
        failed = [r for r in self.results if 'error' in r]
        
        total_memories = sum(r.get('memories', 0) for r in successful)
        total_characters = set()
        for r in successful:
            total_characters.update(r.get('new_characters', []))
        
        report = {
            'summary': {
                'total_files': len(self.results),
                'successful': len(successful),
                'failed': len(failed),
                'total_memories_created': total_memories,
                'unique_new_characters': len(total_characters),
                'average_processing_time': sum(r.get('processing_time', 0) for r in successful) / len(successful) if successful else 0
            },
            'new_characters': list(total_characters),
            'failed_files': [r['file'] for r in failed],
            'detailed_results': self.results
        }
        
        return report

# Usage
def run_batch_processing():
    config = {
        'db_path': 'batch_narrative.db',
        'batch_size': 20
    }
    
    processor = BatchNarrativeProcessor(config)
    report = processor.process_chapter_directory("./narrative_chapters")
    
    # Save report
    with open('processing_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Batch processing complete!")
    print(f"   Files processed: {report['summary']['total_files']}")
    print(f"   Memories created: {report['summary']['total_memories_created']}")
    print(f"   New characters: {len(report['new_characters'])}")
    
    return report
```

## 🎯 Advanced Character Integration

### Custom Character Trait Analysis

```python
from memory_system import NewCharacterIntegrator

class AdvancedCharacterAnalyzer(NewCharacterIntegrator):
    def __init__(self, memory_system):
        super().__init__(memory_system)
        self.trait_patterns = {
            'leadership': [r'lead', r'command', r'direct', r'manage', r'guide'],
            'analytical': [r'analyz', r'calculate', r'deduce', r'logic', r'reason'],
            'manipulative': [r'manipulat', r'control', r'influence', r'scheme'],
            'empathetic': [r'caring', r'understand', r'support', r'comfort'],
            'secretive': [r'secret', r'hidden', r'private', r'conceal'],
            'social': [r'friend', r'group', r'party', r'social', r'network']
        }
    
    def advanced_trait_inference(self, character_name):
        """Advanced personality trait inference with confidence scores"""
        char_memories = [
            m for m in self.memory_system.memories.values()
            if m.character_perspective == character_name
        ]
        
        if not char_memories:
            return {}
        
        trait_scores = {}
        
        # Analyze content patterns
        all_content = ' '.join(m.content.lower() for m in char_memories)
        
        for trait, patterns in self.trait_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, all_content))
                score += matches
            
            # Normalize by content length
            normalized_score = score / len(all_content.split()) * 1000
            trait_scores[trait] = normalized_score
        
        # Add behavioral analysis
        behavioral_traits = self._analyze_behavioral_patterns(char_memories)
        trait_scores.update(behavioral_traits)
        
        # Filter significant traits
        significant_traits = {
            trait: score for trait, score in trait_scores.items()
            if score > 0.5  # Confidence threshold
        }
        
        return significant_traits
    
    def _analyze_behavioral_patterns(self, memories):
        """Analyze behavioral patterns from memory data"""
        if not memories:
            return {}
        
        # Interaction patterns
        avg_participants = np.mean([len(m.participants) for m in memories])
        social_score = min(avg_participants - 1, 3) / 3  # Normalize 0-1
        
        # Emotional patterns
        emotional_weights = [m.emotional_weight for m in memories]
        emotional_variance = np.var(emotional_weights)
        stability_score = 1.0 / (1.0 + emotional_variance)
        
        # Secret involvement
        secret_ratio = sum(1 for m in memories if m.is_secret) / len(memories)
        
        return {
            'social_behavioral': social_score,
            'emotional_stability': stability_score,
            'secretive_behavioral': secret_ratio
        }

# Usage
def analyze_new_character(memory_system, character_name):
    analyzer = AdvancedCharacterAnalyzer(memory_system)
    traits = analyzer.advanced_trait_inference(character_name)
    
    print(f"🎭 Advanced analysis for {character_name}:")
    for trait, score in sorted(traits.items(), key=lambda x: x[1], reverse=True):
        confidence = "High" if score > 1.0 else "Medium" if score > 0.5 else "Low"
        print(f"   {trait}: {score:.3f} ({confidence} confidence)")
    
    return traits
```

## 📊 Production Monitoring & Analytics

### Real-Time Dashboard

```python
import time
from datetime import datetime, timedelta
from memory_system import MemorySystem, ComprehensiveEvaluationPipeline

class NarrativeDashboard:
    def __init__(self, memory_system):
        self.memory_system = memory_system
        self.evaluator = ComprehensiveEvaluationPipeline(memory_system)
        self.metrics_history = []
    
    def collect_metrics(self):
        """Collect current system metrics"""
        stats = self.memory_system.get_narrative_growth_stats()
        
        # Run lightweight evaluation
        evaluation_results = self.evaluator.run_full_evaluation()
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_memories': stats['total_memories'],
            'total_characters': stats['total_characters'],
            'total_chapters': stats['total_chapters'],
            'relationship_count': stats['relationship_network_growth']['total_relationships'],
            'recent_updates': stats['recent_updates'],
            'overall_score': evaluation_results['overall_score'],
            'consistency_score': evaluation_results['question_b_consistency']['overall_score'],
            'retrieval_score': evaluation_results['question_a_retrieval']['overall_score'],
            'character_count_by_chapter': len(stats['character_introduction_timeline'])
        }
        
        self.metrics_history.append(metrics)
        
        # Keep last 100 measurements
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)
        
        return metrics
    
    def generate_dashboard_data(self):
        """Generate data for dashboard visualization"""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        
        # Calculate trends
        if len(self.metrics_history) >= 2:
            previous = self.metrics_history[-2]
            trends = {
                'memory_growth': latest['total_memories'] - previous['total_memories'],
                'character_growth': latest['total_characters'] - previous['total_characters'],
                'score_trend': latest['overall_score'] - previous['overall_score']
            }
        else:
            trends = {'memory_growth': 0, 'character_growth': 0, 'score_trend': 0}
        
        return {
            'current_metrics': latest,
            'trends': trends,
            'time_series': self.metrics_history[-20:],  # Last 20 measurements
            'alerts': self._generate_alerts(latest, trends)
        }
    
    def _generate_alerts(self, current, trends):
        """Generate system alerts"""
        alerts = []
        
        if current['overall_score'] < 0.6:
            alerts.append({
                'level': 'warning',
                'message': f"Low system score: {current['overall_score']:.3f}",
                'recommendation': "Review recent memory additions for quality"
            })
        
        if trends['score_trend'] < -0.1:
            alerts.append({
                'level': 'warning', 
                'message': f"Score declining: {trends['score_trend']:+.3f}",
                'recommendation': "Check memory consistency and resolve contradictions"
            })
        
        if current['recent_updates'] == 0:
            alerts.append({
                'level': 'info',
                'message': "No recent updates detected",
                'recommendation': "Check file watcher or add new content"
            })
        
        return alerts
    
    def export_metrics(self, filename=None):
        """Export metrics to JSON file"""
        if filename is None:
            filename = f"narrative_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'system_info': {
                'total_measurements': len(self.metrics_history),
                'measurement_period': self.metrics_history[0]['timestamp'] if self.metrics_history else None
            },
            'metrics_history': self.metrics_history,
            'dashboard_data': self.generate_dashboard_data()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📊 Metrics exported to {filename}")
        return filename

# Usage
def setup_monitoring(memory_system):
    dashboard = NarrativeDashboard(memory_system)
    
    # Collect metrics every 5 minutes
    while True:
        try:
            metrics = dashboard.collect_metrics()
            dashboard_data = dashboard.generate_dashboard_data()
            
            print(f"📊 {metrics['timestamp']}")
            print(f"   Memories: {metrics['total_memories']} (+{dashboard_data['trends']['memory_growth']})")
            print(f"   Characters: {metrics['total_characters']} (+{dashboard_data['trends']['character_growth']})")
            print(f"   Score: {metrics['overall_score']:.3f} ({dashboard_data['trends']['score_trend']:+.3f})")