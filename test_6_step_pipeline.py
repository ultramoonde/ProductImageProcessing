#!/usr/bin/env python3
"""
Simple 6-Step Modular Pipeline Test Runner
Tests the enhanced pipeline with brand detection and price calculations
"""

import sys
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Add src paths for imports
sys.path.append('src')
sys.path.append('.')

from src.interfaces import StepInput, StepOutput
from src.steps import step_1_ui_analysis
from src.steps import step_2_category_analysis
from src.steps import step_3_canvas_detection
from src.steps import step_4_component_extraction
from src.steps import step_5_consensus_analysis
from src.steps import step_6_csv_generation

def run_6_step_pipeline(image_path: str, output_dir: str):
    """Run the complete 6-step modular pipeline"""

    # Setup
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🧪 6-STEP MODULAR PIPELINE TEST")
    print("=" * 60)
    print(f"📸 Input Image: {image_path}")
    print(f"📁 Output Dir: {output_dir}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return False

    # Initialize pipeline input
    pipeline_input = StepInput(
        image=image,
        image_name=image_path.stem,
        current_image_dir=output_dir
    )

    # Step 1: UI Analysis
    print("\n🔍 STEP 1: UI Analysis")
    result_1 = step_1_ui_analysis.run(pipeline_input)
    if not result_1.success:
        print(f"❌ Step 1 failed: {result_1.data.get('error', 'Unknown error')}")
        return False
    pipeline_input.data['step_1'] = result_1.data
    print(f"✅ Step 1 completed")

    # Step 2: Category Analysis
    print("\n📊 STEP 2: Category Analysis")
    # Pass Step 1 results to Step 2
    pipeline_input.data.update(result_1.data)
    result_2 = step_2_category_analysis.run(pipeline_input)
    if not result_2.success:
        print(f"❌ Step 2 failed: {result_2.data.get('error', 'Unknown error')}")
        return False
    pipeline_input.data['step_2'] = result_2.data
    print(f"✅ Step 2 completed - Categories: {result_2.data.get('main_category', 'N/A')} > {result_2.data.get('active_subcategory', 'N/A')}")

    # Step 3: Canvas Detection
    print("\n🎯 STEP 3: Canvas Detection")
    # Pass Step 2 results to Step 3
    pipeline_input.data.update(result_2.data)
    result_3 = step_3_canvas_detection.run(pipeline_input)
    if not result_3.success:
        print(f"❌ Step 3 failed: {result_3.data.get('error', 'Unknown error')}")
        return False
    pipeline_input.data['step_3'] = result_3.data
    print(f"✅ Step 3 completed - Detected canvas regions")

    # Step 4: Component Extraction
    print("\n📦 STEP 4: Component Extraction")
    # Pass Step 3 results to Step 4
    pipeline_input.data.update(result_3.data)
    result_4 = step_4_component_extraction.run(pipeline_input)
    if not result_4.success:
        print(f"❌ Step 4 failed: {result_4.data.get('error', 'Unknown error')}")
        return False
    pipeline_input.data['step_4'] = result_4.data
    component_count = result_4.data.get('total_components', 0)
    print(f"✅ Step 4 completed - Extracted {component_count} components")

    # Step 5: Consensus Analysis (ENHANCED with brand detection)
    print("\n🤝 STEP 5: Consensus Analysis (Enhanced)")
    # Pass Step 4 results to Step 5
    pipeline_input.data.update(result_4.data)
    result_5 = step_5_consensus_analysis.run(pipeline_input)
    if not result_5.success:
        print(f"❌ Step 5 failed: {result_5.data.get('error', 'Unknown error')}")
        return False
    pipeline_input.data['step_5'] = result_5.data
    analyzed_products = result_5.data.get('analyzed_products', [])
    print(f"✅ Step 5 completed - Analyzed {len(analyzed_products)} products with enhanced brand detection")

    # Step 6: CSV Generation (ENHANCED with price calculations)
    print("\n📊 STEP 6: CSV Generation (Enhanced)")
    # Pass Step 5 results to Step 6
    pipeline_input.data.update(result_5.data)
    result_6 = step_6_csv_generation.run(pipeline_input)
    if not result_6.success:
        print(f"❌ Step 6 failed: {result_6.data.get('error', 'Unknown error')}")
        return False

    csv_path = result_6.output_files.get('final_csv')
    total_products = result_6.data.get('total_products', 0)

    print(f"✅ Step 6 completed - Generated CSV with {total_products} products")
    print(f"📁 Final CSV: {csv_path}")

    # Show sample results
    if csv_path and Path(csv_path).exists():
        print("\n📋 SAMPLE RESULTS (First 3 products):")
        import pandas as pd
        df = pd.read_csv(csv_path)

        # Show key columns
        key_columns = ['product_name', 'brand', 'price', 'weight_quantity', 'unit',
                      'price_per_kg', 'price_per_piece', 'main_category', 'active_subcategory']
        available_columns = [col for col in key_columns if col in df.columns]

        sample_df = df[available_columns].head(3)
        print(sample_df.to_string(index=False))

        print(f"\n📊 Total CSV columns: {len(df.columns)}")
        print(f"📊 New enhanced fields included: price_per_kg, price_per_piece, price_per_liter")

    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    return True

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run 6-step modular pipeline test')
    parser.add_argument('--image', '-i', required=True, help='Input image path')
    parser.add_argument('--output', '-o', default='fresh_test_output', help='Output directory')

    args = parser.parse_args()

    success = run_6_step_pipeline(args.image, args.output)
    sys.exit(0 if success else 1)