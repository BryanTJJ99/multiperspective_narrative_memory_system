# env_config.py
"""
Environment Configuration Module
Manages environment variables and configuration settings for the memory system
"""

import os
from typing import Optional
import logging

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    logging.warning("python-dotenv not installed. Environment variables will be read from system only.")

class EnvironmentConfig:
    """Manages environment configuration with .env file support"""
    
    def __init__(self, env_file: str = ".env"):
        self.env_file = env_file
        self._load_environment()
    
    def _load_environment(self):
        """Load environment variables from .env file if available"""
        if DOTENV_AVAILABLE:
            # Try to load from .env file
            if os.path.exists(self.env_file):
                load_dotenv(self.env_file)
                logging.info(f"Loaded environment variables from {self.env_file}")
            else:
                logging.info(f"No {self.env_file} file found. Using system environment variables.")
        else:
            logging.info("Using system environment variables only.")
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment"""
        return os.getenv('OPENAI_API_KEY')
    
    @property
    def openai_model(self) -> str:
        """Get preferred OpenAI model"""
        return os.getenv('OPENAI_MODEL', 'gpt-4')
    
    @property
    def openai_temperature(self) -> float:
        """Get OpenAI temperature setting"""
        try:
            return float(os.getenv('OPENAI_TEMPERATURE', '0.3'))
        except ValueError:
            return 0.3
    
    @property
    def openai_max_tokens(self) -> int:
        """Get OpenAI max tokens setting"""
        try:
            return int(os.getenv('OPENAI_MAX_TOKENS', '1000'))
        except ValueError:
            return 1000
    
    @property
    def db_path(self) -> str:
        """Get database path"""
        return os.getenv('DB_PATH', 'narrative_memory.db')
    
    @property
    def watch_directory(self) -> Optional[str]:
        """Get watch directory for file monitoring"""
        return os.getenv('WATCH_DIRECTORY')
    
    @property
    def log_level(self) -> str:
        """Get logging level"""
        return os.getenv('LOG_LEVEL', 'INFO')
    
    @property
    def enable_file_watcher(self) -> bool:
        """Check if file watcher should be enabled"""
        return os.getenv('ENABLE_FILE_WATCHER', 'true').lower() in ('true', '1', 'yes', 'on')
    
    @property
    def consolidation_threshold(self) -> int:
        """Get memory consolidation threshold"""
        try:
            return int(os.getenv('CONSOLIDATION_THRESHOLD', '10'))
        except ValueError:
            return 10
    
    @property
    def max_concurrent_api_calls(self) -> int:
        """Get maximum concurrent API calls"""
        try:
            return int(os.getenv('MAX_CONCURRENT_API_CALLS', '5'))
        except ValueError:
            return 5
    
    @property
    def api_rate_limit_per_minute(self) -> int:
        """Get API rate limit per minute"""
        try:
            return int(os.getenv('API_RATE_LIMIT_PER_MINUTE', '60'))
        except ValueError:
            return 60
    
    def validate_configuration(self) -> dict:
        """Validate configuration and return status"""
        status = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check OpenAI API key
        if not self.openai_api_key:
            status['warnings'].append("No OPENAI_API_KEY found. LLM features will be disabled.")
        elif not self.openai_api_key.startswith('sk-'):
            status['warnings'].append("OPENAI_API_KEY format looks incorrect (should start with 'sk-')")
        
        # Check database path
        db_dir = os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.'
        if not os.access(db_dir, os.W_OK):
            status['errors'].append(f"Database directory '{db_dir}' is not writable")
            status['valid'] = False
        
        # Check watch directory
        if self.watch_directory and not os.path.exists(self.watch_directory):
            status['warnings'].append(f"Watch directory '{self.watch_directory}' does not exist")
        
        return status
    
    def print_configuration(self):
        """Print current configuration (hiding sensitive data)"""
        print("Configuration Settings:")
        print("=" * 50)
        
        # OpenAI settings
        api_key_display = f"{self.openai_api_key[:8]}...{self.openai_api_key[-4:]}" if self.openai_api_key else "Not set"
        print(f"OpenAI API Key: {api_key_display}")
        print(f"OpenAI Model: {self.openai_model}")
        print(f"Temperature: {self.openai_temperature}")
        print(f"Max Tokens: {self.openai_max_tokens}")
        
        # System settings
        print(f"Database Path: {self.db_path}")
        print(f"Watch Directory: {self.watch_directory or 'Not set'}")
        print(f"Log Level: {self.log_level}")
        print(f"File Watcher: {'Enabled' if self.enable_file_watcher else 'Disabled'}")
        print(f"Consolidation Threshold: {self.consolidation_threshold}")
        
        # Performance settings
        print(f"Max Concurrent API Calls: {self.max_concurrent_api_calls}")
        print(f"API Rate Limit: {self.api_rate_limit_per_minute}/minute")
        
        print("=" * 50)
    
    def create_sample_env_file(self, filename: str = ".env.example"):
        """Create a sample .env file with documentation"""
        sample_content = """# .env
# Environment variables for Multiperspective Narrative Memory System
# Copy this file to .env and add your actual values

# OpenAI API Configuration (Required for LLM features)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Model preferences
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.3
OPENAI_MAX_TOKENS=1000

# Database Configuration
DB_PATH=narrative_memory.db
WATCH_DIRECTORY=./new_chapters

# System Configuration
LOG_LEVEL=INFO
ENABLE_FILE_WATCHER=true
CONSOLIDATION_THRESHOLD=10

# Performance Settings
MAX_CONCURRENT_API_CALLS=5
API_RATE_LIMIT_PER_MINUTE=60

# Instructions:
# 1. Copy this file to .env
# 2. Replace 'your_openai_api_key_here' with your actual OpenAI API key
# 3. Adjust other settings as needed
# 4. Never commit the .env file to version control
"""
        
        with open(filename, 'w') as f:
            f.write(sample_content)
        
        print(f"Sample environment file created: {filename}")
        print(f"Copy this to .env and add your OpenAI API key")


# Global configuration instance
env_config = EnvironmentConfig()

# Configure logging based on environment
logging.basicConfig(
    level=getattr(logging, env_config.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    # Demonstrate configuration usage
    print("Environment Configuration Demo")
    print("=" * 40)
    
    # Print current configuration
    env_config.print_configuration()
    
    # Validate configuration
    status = env_config.validate_configuration()
    
    print("\nValidation Results:")
    print(f"Valid: {status['valid']}")
    
    if status['warnings']:
        print("Warnings:")
        for warning in status['warnings']:
            print(f"  - {warning}")
    
    if status['errors']:
        print("Errors:")
        for error in status['errors']:
            print(f"  - {error}")
    
    # Create sample file if needed
    if not os.path.exists('.env') and not os.path.exists('.env.example'):
        env_config.create_sample_env_file()