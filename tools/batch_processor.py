#!/usr/bin/env python3
"""
Batch processor for sequential image processing with CSV aggregation.
Processes multiple images through the 6-step pipeline and aggregates results.
"""

import sys
import cv2
from pathlib import Path
import pandas as pd
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

def process_single_image(image_path: Path, output_base_dir: Path):
    """Process a single image through all 6 steps."""
    print(f"\n{'='*80}")
    print(f"🔄 PROCESSING: {image_path.name}")
    print(f"{'='*80}\n")

    # Create output directory for this image
    image_name = image_path.stem
    output_dir = output_base_dir / image_name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return None

        # Initialize pipeline input
        pipeline_input = StepInput(
            image=image,
            image_name=image_name,
            current_image_dir=output_dir
        )

        # Step 1: UI Analysis
        print("🔍 STEP 1: UI Analysis")
        result_1 = step_1_ui_analysis.run(pipeline_input)
        if not result_1.success:
            print(f"❌ Step 1 failed: {result_1.data.get('error', 'Unknown error')}")
            return None
        pipeline_input.data['step_1'] = result_1.data
        print(f"✅ Step 1 completed")

        # Step 2: Category Analysis
        print("📊 STEP 2: Category Analysis")
        pipeline_input.data.update(result_1.data)
        result_2 = step_2_category_analysis.run(pipeline_input)
        if not result_2.success:
            print(f"❌ Step 2 failed: {result_2.data.get('error', 'Unknown error')}")
            return None
        pipeline_input.data['step_2'] = result_2.data
        print(f"✅ Step 2 completed")

        # Step 3: Canvas Detection
        print("🎯 STEP 3: Canvas Detection")
        pipeline_input.data.update(result_2.data)
        result_3 = step_3_canvas_detection.run(pipeline_input)
        if not result_3.success:
            print(f"❌ Step 3 failed: {result_3.data.get('error', 'Unknown error')}")
            return None
        pipeline_input.data['step_3'] = result_3.data
        print(f"✅ Step 3 completed")

        # Step 4: Component Extraction
        print("📦 STEP 4: Component Extraction")
        pipeline_input.data.update(result_3.data)
        result_4 = step_4_component_extraction.run(pipeline_input)
        if not result_4.success:
            print(f"❌ Step 4 failed: {result_4.data.get('error', 'Unknown error')}")
            return None
        pipeline_input.data['step_4'] = result_4.data
        print(f"✅ Step 4 completed")

        # Step 5: Consensus Analysis
        print("🤝 STEP 5: Consensus Analysis")
        pipeline_input.data.update(result_4.data)
        result_5 = step_5_consensus_analysis.run(pipeline_input)
        if not result_5.success:
            print(f"❌ Step 5 failed: {result_5.data.get('error', 'Unknown error')}")
            return None
        pipeline_input.data['step_5'] = result_5.data
        print(f"✅ Step 5 completed")

        # Step 6: CSV Generation
        print("📊 STEP 6: CSV Generation")
        pipeline_input.data.update(result_5.data)
        result_6 = step_6_csv_generation.run(pipeline_input)
        if not result_6.success:
            print(f"❌ Step 6 failed: {result_6.data.get('error', 'Unknown error')}")
            return None

        csv_path = result_6.output_files.get('final_csv')
        print(f"✅ {image_path.name} completed successfully!")
        print(f"   CSV: {csv_path}")

        return csv_path

    except Exception as e:
        print(f"❌ Error processing {image_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def aggregate_csvs(csv_paths: list, output_path: Path):
    """Aggregate all individual CSVs into a single batch CSV."""
    print(f"\n{'='*80}")
    print(f"📊 AGGREGATING {len(csv_paths)} CSV FILES")
    print(f"{'='*80}\n")

    all_data = []

    for csv_path in csv_paths:
        if csv_path and Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            all_data.append(df)
            print(f"   ✓ Loaded {csv_path} ({len(df)} rows)")

    if all_data:
        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)

        # Save to batch file
        combined_df.to_csv(output_path, index=False)
        print(f"\n✅ Aggregated CSV saved: {output_path}")
        print(f"   Total rows: {len(combined_df)}")
        print(f"   Total columns: {len(combined_df.columns)}")

        return output_path
    else:
        print("⚠️  No valid CSV files to aggregate")
        return None

def main():
    """Main batch processing function."""
    # Define test images
    approved_dir = Path("/Users/davemooney/_dev/Flink/Approved")
    test_images = [
        approved_dir / "IMG_8280.PNG",
        approved_dir / "IMG_8290.PNG",
        approved_dir / "IMG_8300.PNG",
        approved_dir / "IMG_8310.PNG"
    ]

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = Path(f"batch_processing_{timestamp}")
    output_base_dir.mkdir(exist_ok=True)

    print(f"\n🚀 BATCH PROCESSING STARTED")
    print(f"📁 Output directory: {output_base_dir}")
    print(f"📸 Images to process: {len(test_images)}\n")

    # Process each image sequentially
    csv_paths = []
    for i, image_path in enumerate(test_images, 1):
        print(f"\n[{i}/{len(test_images)}] Processing {image_path.name}...")
        csv_path = process_single_image(image_path, output_base_dir)
        csv_paths.append(csv_path)

    # Aggregate all CSVs
    batch_csv_path = output_base_dir / "batch_Final.csv"
    aggregate_csvs(csv_paths, batch_csv_path)

    # Open output directory
    import subprocess
    subprocess.run(["open", str(output_base_dir)])

    print(f"\n{'='*80}")
    print(f"✅ BATCH PROCESSING COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
