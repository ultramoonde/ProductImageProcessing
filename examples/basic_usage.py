#!/usr/bin/env python3
"""
Basic Usage Example for ProductImageProcessing Pipeline

This example shows how to use the step-by-step pipeline to process
a single Flink grocery app screenshot.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from step_by_step_pipeline import StepByStepPipeline

def main():
    """Basic example of processing a single image"""

    # Initialize the pipeline
    pipeline = StepByStepPipeline()

    # Example image path (you'll need to provide your own Flink screenshot)
    image_path = "path/to/your/flink_screenshot.png"

    if not os.path.exists(image_path):
        print("❌ Please provide a valid Flink screenshot path")
        print("   Expected format: Flink grocery app screenshot with products grid")
        return

    try:
        # Process the image through all 6 steps
        print(f"🚀 Processing {os.path.basename(image_path)}...")

        results = pipeline.process_image(image_path)

        if results.get("success"):
            print("✅ Processing completed successfully!")
            print(f"📁 Results saved to: {results.get('output_dir')}")
            print(f"📊 Products extracted: {len(results.get('products', []))}")

            # Show what was extracted
            for i, product in enumerate(results.get('products', []), 1):
                print(f"   {i}. {product.get('name', 'Unknown product')}")
        else:
            print(f"❌ Processing failed: {results.get('error')}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()