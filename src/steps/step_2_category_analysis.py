#!/usr/bin/env python3
"""
Step 2: Header Category Analysis
Analyzes screenshot header to detect German food categories using LLM consensus
"""

import sys
import cv2
import numpy as np
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List

# Add project paths for imports
sys.path.append('src')
sys.path.append('.')

from src.interfaces import StepInput, StepOutput, CategoryAnalysisResult
from src.local_consensus_analyzer import LocalConsensusAnalyzer

def run(input_data: StepInput) -> StepOutput:
    """
    Perform header category analysis using LLM consensus

    Args:
        input_data: StepInput with image and UI region data

    Returns:
        StepOutput with detected categories and analysis
    """
    try:
        print("🏷️ STEP 2: Header Category Analysis using consensus system")

        image = input_data.image
        image_name = input_data.image_name
        output_dir = input_data.current_image_dir

        # Get header region from previous step
        header_region = input_data.data.get("header_region")

        if not header_region:
            print("   ❌ No header region provided - skipping category analysis")
            return StepOutput(
                success=False,
                step_name="Category Analysis",
                errors=["No header region provided"],
                data={
                    "status": "error",
                    "category_data": None
                }
            )

        # Extract full header region first
        x, y = header_region['x'], header_region['y']
        width = header_region['width']
        header_height = header_region['height']
        full_header_image = image[y:y+header_height, x:x+width]

        print(f"   📋 Full header region: {width}x{header_height} at ({x}, {y})")

        # Use consensus system for category analysis
        print("   🧠 Using 3-model consensus system for category detection...")

        analyzer = LocalConsensusAnalyzer()
        consensus_result = asyncio.run(
            analyzer.analyze_categories_with_consensus(full_header_image)
        )

        output_files = {}

        # Save header image
        if output_dir:
            header_path = output_dir / f"{image_name}_02_analysis.jpg"
            cv2.imwrite(str(header_path), full_header_image)
            output_files["header_image"] = str(header_path)

        if consensus_result:
            print(f"   📊 Consensus result keys: {list(consensus_result.keys())}")

            # Debug: Check if hierarchy fields exist
            hierarchy_fields = {
                "main_category": consensus_result.get("main_category"),
                "active_subcategory": consensus_result.get("active_subcategory"),
                "available_subcategories": consensus_result.get("available_subcategories"),
                "visual_hierarchy": consensus_result.get("visual_hierarchy")
            }
            print(f"   🏗️ Hierarchy fields found: {hierarchy_fields}")

            # Debug: Check individual results for hierarchy data
            individual_results = consensus_result.get("individual_results", [])
            print(f"   🔍 Individual results count: {len(individual_results)}")
            for i, result in enumerate(individual_results):
                result_data = result.get("data", {})
                print(f"   🔍 Result {i+1} keys: {list(result_data.keys())}")
                if "main_category" in result_data:
                    print(f"   🔍 Result {i+1} hierarchy: main='{result_data.get('main_category')}', active='{result_data.get('active_subcategory')}')")

            # Extract category information directly from consensus result
            main_category = "Unknown"
            active_subcategory = "Unknown"
            available_subcategories = []

            # Extract hierarchy data that LLM consensus already provides
            # First try to get the proper hierarchy structure
            main_category = consensus_result.get("main_category", "")
            active_subcategory = consensus_result.get("active_subcategory", "")
            available_subcategories = consensus_result.get("available_subcategories", [])

            # Always get categories field for all_detected_categories
            categories_field = consensus_result.get("categories", [])

            # If hierarchy fields are empty, fall back to legacy parsing
            if not main_category and not active_subcategory:
                current_field = consensus_result.get("current", "")

                if isinstance(categories_field, list) and categories_field:
                    available_subcategories = categories_field
                    if current_field and current_field in categories_field:
                        main_category = current_field
                        active_subcategory = current_field
                    else:
                        # Use the first category as fallback
                        main_category = categories_field[0]
                        active_subcategory = categories_field[0]

            category_data = {
                'main_category': main_category,
                'active_subcategory': active_subcategory,
                'available_subcategories': available_subcategories,
                'confidence': 0.8,
                'method': 'consensus_ui_analysis',
                'all_detected_categories': categories_field
            }

            print(f"   ✅ Category detected: {category_data['main_category']}")
            print(f"   📊 Subcategory: {category_data['active_subcategory']}")
        else:
            print("   ⚠️ Consensus analysis failed - using fallback")
            category_data = {
                'main_category': 'Unknown',
                'active_subcategory': 'Unknown',
                'available_subcategories': [],
                'confidence': 0.0,
                'method': 'fallback'
            }

        # Save category analysis
        if output_dir:
            analysis_path = output_dir / f"{image_name}_02_analysis.json"
            with open(analysis_path, 'w') as f:
                json.dump(category_data, f, indent=2)
            output_files["analysis_json"] = str(analysis_path)

        return StepOutput(
            success=True,
            step_name="Category Analysis",
            data={
                "status": "success",
                "category_data": category_data,
                "main_category": category_data.get('main_category'),
                "active_subcategory": category_data.get('active_subcategory'),
                "available_subcategories": category_data.get('available_subcategories', []),
                "all_detected_categories": category_data.get('all_detected_categories', []),
                "confidence": category_data.get('confidence', 0.0),
                "method": category_data.get('method')
            },
            output_files=output_files
        )

    except Exception as e:
        print(f"   ❌ Header category analysis failed: {e}")
        return StepOutput(
            success=False,
            step_name="Category Analysis",
            errors=[f"Category analysis failed: {str(e)}"],
            data={
                "status": "error",
                "category_data": None
            }
        )


def _fallback_category_extraction(text: str, filename: str) -> Dict[str, Any]:
    """Fallback category extraction using text patterns"""
    text_lower = text.lower()

    # Expanded category patterns
    category_patterns = {
        "Obst": ["obst", "fruit", "früchte"],
        "Gemüse": ["gemüse", "vegetables", "vegetable"],
        "Joghurt & Desserts": ["joghurt", "dessert", "yogurt", "pudding", "quark"],
        "Milch & Butter": ["milch", "butter", "käse", "sahne", "milk", "cheese"],
        "Backwaren": ["backwaren", "brot", "bakery", "bread"],
        "Kinder": ["kinder", "baby", "kids", "children"],
        "Fleisch & Wurst": ["fleisch", "wurst", "meat", "sausage", "schinken"],
        "Tiefkühl": ["tiefkühl", "frozen", "tk", "gefroren"],
        "Getränke": ["getränke", "drinks", "wasser", "saft", "beverages"]
    }

    detected_category = "Unknown"
    for category, patterns in category_patterns.items():
        if any(pattern in text_lower for pattern in patterns):
            detected_category = category
            break

    return {
        "main_category": detected_category,
        "active_subcategory": "Unknown",
        "available_subcategories": [],
        "navigation_text": [text[:100]],
        "confidence": 0.6,
        "method": "text_fallback"
    }