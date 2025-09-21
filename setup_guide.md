# Environment Setup Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
# Option A: Create .env file (Recommended)
cp .env.example .env
# Edit .env and add your OpenAI API key

# Option B: Set environment variable
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 3. Run the System
```bash
python complete_example.py
```

## Detailed Setup

### Environment File (.env)
The system uses a `.env` file to securely store configuration. Create a `.env` file in the project root:

```env
# Required for LLM features
OPENAI_API_KEY=sk-your-actual-openai-api-key-here

# Optional configurations
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.3
OPENAI_MAX_TOKENS=1000

DB_PATH=narrative_memory.db
WATCH_DIRECTORY=./new_chapters
LOG_LEVEL=INFO
ENABLE_FILE_WATCHER=true
CONSOLIDATION_THRESHOLD=10
```

### Security
- **Never commit `.env` files** - they're in `.gitignore`
- The system gracefully falls back to statistical analysis if no API key is provided
- API keys are masked in log output for security

### Configuration Validation
The system automatically validates your configuration on startup:

```python
from env_config import env_config

# Print current configuration (API key masked)
env_config.print_configuration()

# Validate settings
status = env_config.validate_configuration()
print(f"Configuration valid: {status['valid']}")
```

### File Structure
```
multiperspective_narrative_memory_system/
├── .env                          # Your environment variables (DO NOT COMMIT)
├── .env.example                  # Template file (safe to commit)
├── .gitignore                    # Protects sensitive files
├── env_config.py                 # Environment configuration module
├── dynamic_memory_system.py      # Enhanced with env config
├── data_loader.py               # Enhanced with env config
├── complete_example.py          # Enhanced with env config
├── memory_system.py             # Updated imports
├── requirements.txt             # Includes python-dotenv
└── [other files...]
```

### Usage Patterns

#### Basic Usage (Automatic Configuration)
```python
from memory_system import MemorySystem, MemoryDataProcessor

# Uses .env file automatically
memory_system = MemorySystem()
processor = MemoryDataProcessor(memory_system)
```

#### Manual Override
```python
# Override environment settings if needed
memory_system = MemorySystem(
    db_path="custom_db.db",
    openai_api_key="custom-key"  # Not recommended
)
```

#### Cost Control
```python
# Use cheaper model in .env file
OPENAI_MODEL=gpt-3.5-turbo
```

### Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | None | For LLM features |
| `OPENAI_MODEL` | Model to use | gpt-4 | No |
| `OPENAI_TEMPERATURE` | Model temperature | 0.3 | No |
| `OPENAI_MAX_TOKENS` | Max tokens per request | 1000 | No |
| `DB_PATH` | Database file path | narrative_memory.db | No |
| `WATCH_DIRECTORY` | Auto-watch directory | None | No |
| `LOG_LEVEL` | Logging level | INFO | No |
| `ENABLE_FILE_WATCHER` | Enable file monitoring | true | No |
| `CONSOLIDATION_THRESHOLD` | Memory consolidation trigger | 10 | No |

### Troubleshooting

#### No LLM Features
If you see "LLM features disabled":
1. Check if `.env` file exists
2. Verify `OPENAI_API_KEY` is set correctly
3. Ensure API key starts with `sk-`
4. Check if `python-dotenv` is installed

#### Configuration Issues
```python
# Debug configuration
from env_config import env_config
env_config.print_configuration()
status = env_config.validate_configuration()
print(status)
```

#### Create Sample .env
```python
from env_config import env_config
env_config.create_sample_env_file()
```

### Security Best Practices

1. **Never commit `.env` files**
2. **Use environment variables in production**
3. **Rotate API keys regularly**
4. **Set appropriate file permissions** (`chmod 600 .env`)
5. **Use separate keys for development/production**

### Production Deployment

For production, set environment variables directly instead of using `.env`:

```bash
# Docker
ENV OPENAI_API_KEY=your-key-here

# Kubernetes
apiVersion: v1
kind: Secret
metadata:
  name: openai-secret
data:
  OPENAI_API_KEY: <base64-encoded-key>

# Traditional server
export OPENAI_API_KEY="your-key-here"
systemctl restart your-service
```

The system will automatically use environment variables when no `.env` file is present.