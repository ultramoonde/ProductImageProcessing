#!/usr/bin/env python3
"""
Vision API Configuration
Add your API keys here for real vision analysis
"""

# Option 1: Claude API (Anthropic)
CLAUDE_API_KEY = None  # Set via environment variable ANTHROPIC_API_KEY
# Get from: https://console.anthropic.com/

# Option 2: Gemini API (Google)
GEMINI_API_KEY = None  # Add your Gemini API key here
# Get from: https://makersuite.google.com/app/apikey

# You can also set these as environment variables:
# export ANTHROPIC_API_KEY="your_claude_key"
# export GOOGLE_API_KEY="your_gemini_key"

def get_vision_api_key():
    """Get the first available API key"""
    import os
    
    # Try environment variables first
    claude_key = os.getenv('ANTHROPIC_API_KEY') or os.getenv('CLAUDE_API_KEY') or CLAUDE_API_KEY
    gemini_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY') or GEMINI_API_KEY
    
    if claude_key:
        return claude_key, "claude"
    elif gemini_key:
        return gemini_key, "gemini"
    else:
        return None, None

def get_api_status():
    """Check which APIs are available"""
    api_key, model = get_vision_api_key()
    
    if api_key:
        return f"✅ {model.upper()} API configured"
    else:
        return "⚠️  No vision API keys configured - using mock analysis"