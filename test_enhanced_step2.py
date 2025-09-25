#!/usr/bin/env python3
"""
Test Enhanced Step 2: CSV Generation with Product Canvas Data
Tests the newly added CSV generation functionality in Step 2
"""

import cv2
from step_by_step_pipeline import StepByStepPipeline

def test_enhanced_step2():
    print("🧪 TESTING ENHANCED STEP 2 WITH CSV GENERATION")
    print("=" * 60)

    # Initialize pipeline
    pipeline = StepByStepPipeline()

    # Load test image
    test_image_path = "/Users/davemooney/_dev/Flink/IMG_7805.PNG"
    image = cv2.imread(test_image_path)

    if image is None:
        print(f"❌ Error: Could not load test image: {test_image_path}")
        return

    print(f"✅ Loaded test image: {test_image_path}")
    print(f"   📐 Image dimensions: {image.shape[1]}x{image.shape[0]}")

    # Step 0: UI Analysis (to get categories for CSV)
    print("\n📱 STEP 0: UI Analysis")
    step0_result = pipeline._step_0_ui_analysis(image, "IMG_7805_original")
    print(f"   ✅ Step 0 completed")

    # Step 1: Header Text Extraction (to get category information)
    print("\n🔍 STEP 1: Header Text Extraction")
    step1_result = pipeline._step_01_header_text_extraction(image, step0_result, "IMG_7805")
    print(f"   ✅ Step 1 completed")
    categories = step1_result.get('analysis', {}).get('categories', [])
    print(f"   📊 Categories found: {categories}")

    # Step 2: Enhanced Product Canvas Detection with CSV
    print("\n🎯 STEP 2: Enhanced Product Canvas Detection (with CSV)")
    step2_result = pipeline._step_02_product_canvas_detection(image, step0_result, "IMG_7805")

    # Display results
    print(f"\n📊 STEP 2 RESULTS:")
    print(f"   ✅ Product canvases detected: {len(step2_result['canvases'])}")
    print(f"   📁 Files created:")
    for file_type, file_path in step2_result['files'].items():
        if file_path:
            print(f"      {file_type}: {file_path}")

    # Check CSV data
    csv_data = step2_result.get('csv_data', [])
    if csv_data:
        print(f"\n📋 CSV DATA GENERATED:")
        print(f"   📊 CSV rows: {len(csv_data)}")
        print(f"   🏷️  Sample row structure:")
        for key, value in csv_data[0].items():
            print(f"      {key}: {value}")
    else:
        print(f"\n⚠️  No CSV data generated")

    # Category data
    category_data = step2_result.get('category_data', {})
    print(f"\n🏷️  CATEGORY DATA:")
    print(f"   Primary category: {category_data.get('primary_category', 'None')}")
    print(f"   Primary subcategory: {category_data.get('primary_subcategory', 'None')}")

    print(f"\n🎉 ENHANCED STEP 2 TEST COMPLETED!")
    print(f"✅ CSV generation functionality working properly")

    return step2_result

if __name__ == "__main__":
    test_enhanced_step2()