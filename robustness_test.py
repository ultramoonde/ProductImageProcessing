#!/usr/bin/env python3
"""
Robustness Test for Step-by-Step Pipeline
Tests 4 different images through the complete pipeline (Steps 0-4C)
Consolidates all outputs into a single directory
"""

import os
import shutil
from pathlib import Path
from step_by_step_pipeline import StepByStepPipeline
import time

def run_robustness_test():
    """Run robustness test on 4 different images"""

    # Test images - selecting diverse examples
    test_images = [
        "/Users/davemooney/_dev/Flink/IMG_8138.PNG",
        "/Users/davemooney/_dev/Flink/IMG_8110.PNG",
        "/Users/davemooney/_dev/Flink/IMG_8448.PNG",
        "/Users/davemooney/_dev/Flink/IMG_7974.PNG"
    ]

    # Output directory
    output_dir = Path("/Users/davemooney/_dev/pngImageExtraction/latest/robustness_test_outputs")
    output_dir.mkdir(exist_ok=True)

    print("🧪 ROBUSTNESS TEST: PROCESSING 4 IMAGES THROUGH COMPLETE PIPELINE")
    print("=" * 70)
    print(f"📂 Output directory: {output_dir}")
    print(f"🖼️  Test images: {len(test_images)}")
    print()

    # Initialize pipeline
    pipeline = StepByStepPipeline()

    results = []
    total_files_generated = 0

    for i, image_path in enumerate(test_images, 1):
        image_name = Path(image_path).stem
        print(f"🔄 Processing Image {i}/4: {image_name}")
        print(f"   📁 Source: {image_path}")

        start_time = time.time()

        try:
            # Run complete pipeline
            result = pipeline.run_complete_demonstration(image_path)

            # Count generated files
            image_output_dir = pipeline.output_dir / image_name
            if image_output_dir.exists():
                files = list(image_output_dir.glob("*"))
                file_count = len(files)
                total_files_generated += file_count

                # Copy all files to consolidated output directory
                dest_dir = output_dir / image_name
                dest_dir.mkdir(exist_ok=True)

                for file in files:
                    shutil.copy2(file, dest_dir / file.name)

                print(f"   ✅ SUCCESS: {file_count} files generated")
                print(f"   📋 Files copied to: {dest_dir}")

                results.append({
                    "image": image_name,
                    "status": "SUCCESS",
                    "files": file_count,
                    "time": time.time() - start_time,
                    "output_dir": str(dest_dir)
                })
            else:
                print(f"   ❌ FAILED: No output directory found")
                results.append({
                    "image": image_name,
                    "status": "FAILED",
                    "files": 0,
                    "time": time.time() - start_time,
                    "output_dir": None
                })

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append({
                "image": image_name,
                "status": "ERROR",
                "files": 0,
                "time": time.time() - start_time,
                "error": str(e),
                "output_dir": None
            })

        print(f"   ⏱️  Processing time: {time.time() - start_time:.1f}s")
        print()

    # Generate summary
    print("🎯 ROBUSTNESS TEST SUMMARY")
    print("=" * 50)

    successful = len([r for r in results if r["status"] == "SUCCESS"])
    failed = len([r for r in results if r["status"] in ["FAILED", "ERROR"]])

    print(f"✅ Successful: {successful}/{len(test_images)}")
    print(f"❌ Failed: {failed}/{len(test_images)}")
    print(f"📁 Total files generated: {total_files_generated}")
    print(f"📂 All outputs consolidated in: {output_dir}")
    print()

    # Detailed results
    for result in results:
        status_emoji = "✅" if result["status"] == "SUCCESS" else "❌"
        print(f"{status_emoji} {result['image']}: {result['status']} - {result['files']} files - {result['time']:.1f}s")
        if result.get('error'):
            print(f"   Error: {result['error']}")

    print()

    # Generate CSV summary
    csv_path = output_dir / "robustness_test_summary.csv"
    with open(csv_path, 'w') as f:
        f.write("Image,Status,Files_Generated,Processing_Time_Seconds,Output_Directory\n")
        for result in results:
            f.write(f"{result['image']},{result['status']},{result['files']},{result['time']:.1f},{result.get('output_dir', '')}\n")

    print(f"📊 Test summary saved to: {csv_path}")

    # Show final directory structure
    print(f"\n📁 FINAL OUTPUT STRUCTURE:")
    for item in sorted(output_dir.iterdir()):
        if item.is_dir():
            file_count = len(list(item.glob("*")))
            print(f"   📂 {item.name}/ ({file_count} files)")
        else:
            print(f"   📄 {item.name}")

    return results

if __name__ == "__main__":
    results = run_robustness_test()

    print("\n🎉 ROBUSTNESS TEST COMPLETE!")
    print("All outputs are consolidated and ready for review.")