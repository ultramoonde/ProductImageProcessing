#!/usr/bin/env python3
"""
Test script for the sophisticated Step 0 UI Analysis method
Compares the original Step 0 with the new sophisticated ScreenshotUIAnalyzer version
"""

import cv2
import numpy as np
from step_by_step_pipeline import StepByStepPipeline

def main():
    print("🧪 TESTING SOPHISTICATED STEP 0 vs ORIGINAL STEP 0")
    print("=" * 60)

    # Load test image
    image_path = '/Users/davemooney/_dev/Flink/IMG_7805.PNG'
    print(f"📁 Loading test image: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load test image")
        return

    print(f"✅ Image loaded: {image.shape}")

    # Initialize pipeline
    print("🔧 Initializing pipeline...")
    pipeline = StepByStepPipeline()
    print("✅ Pipeline initialized")

    print("\n" + "="*60)
    print("📱 TESTING SOPHISTICATED STEP 0 (ScreenshotUIAnalyzer)")
    print("="*60)

    # Test the sophisticated Step 0 method
    try:
        sophisticated_result = pipeline._step_0_ui_analysis(image, 'IMG_7805_sophisticated')

        print("\n🎯 SOPHISTICATED STEP 0 RESULTS:")
        print(f"✅ Regions detected: {len(sophisticated_result['regions'])}")
        print(f"✅ Annotated image: {sophisticated_result['annotated_image_path']}")
        print(f"✅ Regions JSON: {sophisticated_result['regions_json_path']}")

        for region_name, region_data in sophisticated_result['regions'].items():
            coords = region_data['coordinates']
            print(f"   📍 {region_name}: [{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}]")

        print("✅ Sophisticated Step 0 completed successfully!")

    except Exception as e:
        print(f"❌ Sophisticated Step 0 failed: {e}")

    print("\n" + "="*60)
    print("🔧 TESTING ORIGINAL STEP 0 (Existing method)")
    print("="*60)

    # Test the original Step 0 method
    try:
        original_result = pipeline._step_00_shop_category_discovery(image, 'IMG_7805_original')

        print("\n🎯 ORIGINAL STEP 0 RESULTS:")
        print(f"✅ Analysis completed")
        print(f"✅ Method: {type(original_result)}")

        # Print any relevant info from original result
        if isinstance(original_result, dict):
            for key, value in original_result.items():
                if not key.startswith('_'):  # Skip private keys
                    print(f"   📊 {key}: {value}")

        print("✅ Original Step 0 completed successfully!")

    except Exception as e:
        print(f"❌ Original Step 0 failed: {e}")

    print("\n" + "="*60)
    print("🎉 COMPARISON COMPLETE!")
    print("="*60)
    print("✅ Both Step 0 implementations have been tested")
    print("📁 Check the step_by_step_demo/ directory for output files")
    print("💡 The sophisticated version uses ScreenshotUIAnalyzer with fallback logic")
    print("💡 The original version uses the existing category discovery system")

if __name__ == "__main__":
    main()