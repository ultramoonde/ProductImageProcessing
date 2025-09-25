"""
Configuration management for food extractor
Loads environment variables from .env file
"""

import os
from pathlib import Path
from typing import Optional

def load_env_file(env_path: Optional[str] = None):
    """Load environment variables from .env file"""
    
    if env_path is None:
        # Look for .env file in current directory, parent directory, etc.
        current_dir = Path(__file__).parent
        for check_dir in [current_dir, current_dir.parent, current_dir.parent.parent]:
            env_file = check_dir / '.env'
            if env_file.exists():
                env_path = str(env_file)
                break
    
    if env_path and os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        print("⚠️ No .env file found - using system environment variables")

def get_supabase_config() -> tuple:
    """Get Supabase configuration from environment"""
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_ANON_KEY')
    
    if not url or not key:
        raise ValueError(
            "Supabase configuration missing. Please set SUPABASE_URL and SUPABASE_ANON_KEY "
            "in your environment variables or create a .env file."
        )
    
    return url, key

# Auto-load environment on import
load_env_file()