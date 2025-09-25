#!/usr/bin/env python3
"""
Nutritionix API Configuration
Real API credentials for comprehensive enrichment
"""

# Your actual Nutritionix API credentials
NUTRITIONIX_CONFIG = {
    'app_id': 'dcdfbe92',
    'api_key': '5f55731309a134421f78436a0e889b3d',
    'base_url': 'https://trackapi.nutritionix.com'
}

def get_nutritionix_headers():
    """Get headers for Nutritionix API requests"""
    return {
        'x-app-id': NUTRITIONIX_CONFIG['app_id'],
        'x-app-key': NUTRITIONIX_CONFIG['api_key'],
        'Content-Type': 'application/json'
    }

def get_nutritionix_config():
    """Get full Nutritionix configuration"""
    return NUTRITIONIX_CONFIG