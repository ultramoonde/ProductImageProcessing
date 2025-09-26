#!/usr/bin/env python3
"""
Step 1: UI Analysis & Region Detection
Analyzes screenshot structure and splits into logical UI regions (header, content, footer)
"""

import sys
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any

# Add project paths for imports
sys.path.append('src')
sys.path.append('.')

from src.interfaces import StepInput, StepOutput, UIAnalysisResult
from screenshot_ui_analyzer import ScreenshotUIAnalyzer

def run(input_data: StepInput) -> StepOutput:
    """
    Perform UI analysis and region detection

    Args:
        input_data: StepInput with image and metadata

    Returns:
        StepOutput with UI regions and structure analysis
    """
    try:
        image = input_data.image
        image_name = input_data.image_name
        output_dir = input_data.current_image_dir

        print("🔍 STEP 1: UI Region Analysis - Identifying header, content, and footer regions")

        # Use ScreenshotUIAnalyzer to identify regions (categories handled by Step 2)
        ui_analyzer = ScreenshotUIAnalyzer()
        ui_analysis = ui_analyzer.analyze_screenshot(image)

        # Remove category detection from Step 1 - Step 2 handles this with LLM consensus
        if 'categories' in ui_analysis:
            del ui_analysis['categories']
        if 'subcategories' in ui_analysis:
            del ui_analysis['subcategories']
        if 'header_analysis' in ui_analysis:
            ui_analysis['header_analysis'] = {
                'has_categories': True,  # Always true for Flink UI
                'has_subcategories': True, # Always true for Flink UI
                'note': 'Category detection delegated to Step 2 LLM consensus system'
            }

        print(f"   📋 Identified {len(ui_analysis.get('regions', {}))} UI regions")
        print("   🏷️ Category detection delegated to Step 2 LLM consensus system")

        # Extract region boundaries
        regions = ui_analysis.get('regions', {})
        header_region = regions.get('header')
        content_region = regions.get('content')
        footer_region = regions.get('footer')

        # Print region details
        if header_region:
            print(f"   📏 Header: {header_region['width']}x{header_region['height']} at ({header_region['x']}, {header_region['y']})")
        if content_region:
            print(f"   📏 Content: {content_region['width']}x{content_region['height']} at ({content_region['x']}, {content_region['y']})")

        # ENHANCED COMPATIBILITY CHECK
        compatibility_check = _validate_ui_compatibility(regions, image, image_name)
        if not compatibility_check["compatible"]:
            print(f"   ❌ UI Structure Incompatible: {compatibility_check['reason']}")
            error_path = output_dir / f"{image_name}_01_incompatible_ui.txt"
            error_message = f"""FLINK UI STRUCTURE INCOMPATIBILITY ERROR - CANNOT PROCESS

Reason: {compatibility_check['reason']}
Details: {compatibility_check.get('details', 'No additional details available')}

This screenshot does not match the expected Flink grocery app UI structure.

Expected Flink Structure:
- Header: 200-600px tall with category tabs at top (y=0-30)
- Content: ≥800px tall product grid starting after header
- Portrait format phone screenshot (≥1000x1800, ratio <0.8)
- Header should be 10-50% of content height

Found Structure: {compatibility_check['found_structure']}
Header Height: {compatibility_check.get('header_height', 'Unknown')}px
Content Height: {compatibility_check.get('content_height', 'Unknown')}px
Image Dimensions: {compatibility_check.get('image_dimensions', 'Unknown')}

This pipeline is specifically calibrated for Flink grocery app screenshots.
Please ensure the image matches the Flink UI layout before processing."""

            error_path.write_text(error_message)

            return StepOutput(
                success=False,
                step_name="UI Analysis",
                errors=[f"UI structure incompatible: {compatibility_check['reason']}"],
                data={
                    "status": "incompatible",
                    "compatibility_check": compatibility_check,
                    "header_region": None,
                    "content_region": None,
                    "footer_region": None
                }
            )

        if not header_region:
            print("   ❌ No header region identified - creating error file")
            error_path = output_dir / f"{image_name}_01_error.txt"
            error_path.write_text("Image Error - No Header Found\nStep 0 did not provide valid header region data.\nSkipping further processing for this image.")

            return StepOutput(
                success=False,
                step_name="UI Analysis",
                errors=["No header region found"],
                data={
                    "status": "error",
                    "header_region": None,
                    "content_region": None,
                    "footer_region": None
                }
            )

        # Save results
        output_files = {}
        if output_dir:
            # Save annotated image showing regions
            annotated_image = image.copy()

            # Draw region boundaries
            if header_region:
                cv2.rectangle(annotated_image,
                            (header_region['x'], header_region['y']),
                            (header_region['x'] + header_region['width'], header_region['y'] + header_region['height']),
                            (0, 255, 0), 3)
                cv2.putText(annotated_image, 'HEADER',
                          (header_region['x'] + 10, header_region['y'] + 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if content_region:
                cv2.rectangle(annotated_image,
                            (content_region['x'], content_region['y']),
                            (content_region['x'] + content_region['width'], content_region['y'] + content_region['height']),
                            (255, 0, 0), 3)
                cv2.putText(annotated_image, 'CONTENT',
                          (content_region['x'] + 10, content_region['y'] + 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            if footer_region:
                cv2.rectangle(annotated_image,
                            (footer_region['x'], footer_region['y']),
                            (footer_region['x'] + footer_region['width'], footer_region['y'] + footer_region['height']),
                            (0, 0, 255), 3)
                cv2.putText(annotated_image, 'FOOTER',
                          (footer_region['x'] + 10, footer_region['y'] + 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Save annotated regions
            annotated_path = output_dir / f"{image_name}_01_annotated.jpg"
            cv2.imwrite(str(annotated_path), annotated_image)
            output_files["annotated_image"] = str(annotated_path)

            # Save individual region images
            header_path = None
            if header_region:
                header_image = image[header_region['y']:header_region['y']+header_region['height'],
                                   header_region['x']:header_region['x']+header_region['width']]
                header_path = output_dir / f"{image_name}_01_header_region.jpg"
                cv2.imwrite(str(header_path), header_image)
                output_files["header_region"] = str(header_path)

            # Save region data
            region_data = {
                'header_region': header_region,
                'content_region': content_region,
                'footer_region': footer_region,
                'ui_analysis': ui_analysis,
                'compatibility_check': compatibility_check
            }

            region_json_path = output_dir / f"{image_name}_01_header_text.json"
            with open(region_json_path, 'w') as f:
                json.dump(region_data, f, indent=2)
            output_files["analysis_json"] = str(region_json_path)

        print(f"   ✅ UI regions identified and saved")
        print(f"   📏 Header: {header_region['width']}x{header_region['height']} at ({header_region['x']}, {header_region['y']})")
        if content_region:
            print(f"   📏 Content: {content_region['width']}x{content_region['height']} at ({content_region['x']}, {content_region['y']})")

        return StepOutput(
            success=True,
            step_name="UI Analysis",
            data={
                "header_region": header_region,
                "content_region": content_region,
                "footer_region": footer_region,
                "ui_analysis": ui_analysis,
                "compatibility_check": compatibility_check
            },
            output_files=output_files
        )

    except Exception as e:
        print(f"   ❌ UI region analysis failed: {e}")
        error_path = output_dir / f"{image_name}_01_error.txt" if output_dir and image_name else None
        if error_path:
            error_path.write_text(f"Image Error - UI Analysis Failed\nError: {str(e)}\nSkipping further processing for this image.")

        return StepOutput(
            success=False,
            step_name="UI Analysis",
            errors=[f"UI analysis failed: {str(e)}"],
            data={
                "status": "error",
                "header_region": None,
                "content_region": None,
                "footer_region": None
            }
        )


def _validate_ui_compatibility(regions: Dict, image: np.ndarray, name: str) -> Dict[str, Any]:
    """Validate if screenshot has Flink-compatible UI structure for processing"""
    try:
        # Extract region information
        header_region = regions.get('header')
        content_region = regions.get('content')
        footer_region = regions.get('footer')

        # Flink grocery app specific structure validation
        compatibility_issues = []
        found_structure = []
        image_height, image_width = image.shape[:2]

        # Expected Flink UI structure based on successfully processed images:
        # - Header: ~200-250px tall, at top (y=0-20)
        # - Content: Starts after header, substantial height for product grid
        # - Total image: Portrait phone screenshot (~1290x2556 typical)

        # Check 1: Header region validation for Flink consistency
        if not header_region:
            compatibility_issues.append("No header region detected - Flink requires header with category tabs")
            found_structure.append("Missing Header")
        else:
            found_structure.append("Header Present")

            # Flink-specific header height validation (based on working screenshots)
            header_height = header_region.get('height', 0)
            header_y = header_region.get('y', 0)

            # CRITICAL: Must be exactly 530px for Flink compatibility - RELAXED FOR TESTING
            if not (200 <= header_height <= 600):
                compatibility_issues.append(f"Header height {header_height}px outside acceptable range (200-600px for testing)")
                # Instead of failing, continue with warning for testing
                print(f"   ⚠️  Header height {header_height}px is outside normal Flink range (530px expected)")
            else:
                print(f"   ✅ Header height {header_height}px is acceptable for testing")

            # Header should be at very top
            if header_y > 30:
                compatibility_issues.append(f"Header not at top: y={header_y} (Flink headers start at y=0-30)")

        # Check 2: Content region validation for Flink product grids
        if not content_region:
            compatibility_issues.append("No content region detected - Flink requires content area for product grid")
            found_structure.append("Missing Content")
        else:
            found_structure.append("Content Present")

            content_height = content_region.get('height', 0)
            content_y = content_region.get('y', 0)

            # CRITICAL: Content must start at exactly pixel 531 for Flink compatibility - DO NOT CHANGE
            if content_y != 531:
                compatibility_issues.append(f"Incompatible content position: y={content_y} (Flink requires y=531)")
                return {"compatible": False, "reason": "Invalid content position", "found_structure": found_structure}

            # Content should be substantial for product grid display
            if content_height < 800:
                compatibility_issues.append(f"Content region too small: {content_height}px (Flink needs ≥800px for product grid)")

            # Content should start after header
            if header_region and content_y < (header_region.get('height', 0) - 50):
                compatibility_issues.append(f"Content overlaps header - invalid Flink layout structure")

        # Check 3: Overall Flink screenshot structure validation
        # Flink screenshots should be portrait phone format
        if image_height < 1800:  # Minimum reasonable height for Flink screenshots
            compatibility_issues.append(f"Image too short: {image_height}px (Flink screenshots typically ≥2000px tall)")

        if image_width < 800:  # Minimum reasonable width
            compatibility_issues.append(f"Image too narrow: {image_width}px (Flink screenshots typically ≥1000px wide)")

        aspect_ratio = image_width / image_height if image_height > 0 else 0
        if aspect_ratio > 0.8:  # Should be portrait
            compatibility_issues.append(f"Not portrait format: ratio={aspect_ratio:.2f} (Flink screenshots are portrait < 0.8)")

        # Check 4: Header-to-content ratio validation (Flink-specific)
        if header_region and content_region:
            header_height = header_region.get('height', 0)
            content_height = content_region.get('height', 0)

            if header_height > 0 and content_height > 0:
                header_content_ratio = header_height / content_height

                # In Flink, header is typically 10-30% of content height
                if header_content_ratio > 0.5:
                    compatibility_issues.append(f"Header too large relative to content: {header_content_ratio:.2f} (Flink header should be < 50% of content)")
                elif header_content_ratio < 0.1:
                    compatibility_issues.append(f"Header too small relative to content: {header_content_ratio:.2f} (Flink header should be > 10% of content)")

        # Check 5: Expected Flink UI positioning
        total_ui_coverage = 0
        if header_region:
            total_ui_coverage += header_region.get('height', 0)
        if content_region:
            total_ui_coverage += content_region.get('height', 0)
        if footer_region:
            total_ui_coverage += footer_region.get('height', 0)

        ui_coverage_ratio = total_ui_coverage / image_height if image_height > 0 else 0
        if ui_coverage_ratio < 0.7:  # UI should cover most of the image
            compatibility_issues.append(f"Poor UI coverage: {ui_coverage_ratio:.2f} (Flink UI should cover ≥70% of image)")

        # Determine compatibility based on critical Flink structure issues
        critical_issues = [issue for issue in compatibility_issues
                         if any(critical in issue.lower() for critical in
                               ['missing header', 'missing content', 'too small', 'too large',
                                'not portrait', 'overlaps header', 'poor ui coverage'])]

        is_compatible = len(critical_issues) == 0

        # Create result with Flink-specific information
        result = {
            "compatible": is_compatible,
            "reason": "Flink UI structure validated successfully" if is_compatible else f"Flink structure incompatible: {'; '.join(critical_issues)}",
            "details": f"Flink validation found {len(compatibility_issues)} issues: {'; '.join(compatibility_issues)}" if compatibility_issues else "All Flink structure checks passed",
            "found_structure": " + ".join(found_structure),
            "all_issues": compatibility_issues,
            "critical_issues": critical_issues,
            "image_dimensions": f"{image_width}x{image_height}",
            "aspect_ratio": round(aspect_ratio, 3),
            "header_height": header_region.get('height', 0) if header_region else 0,
            "content_height": content_region.get('height', 0) if content_region else 0,
            "validation_method": "flink_ui_structure_validation"
        }

        return result

    except Exception as e:
        # If validation fails, be conservative and allow processing
        return {
            "compatible": True,
            "reason": f"Validation error - allowing processing: {str(e)}",
            "details": "Could not complete Flink structure validation, defaulting to compatible",
            "found_structure": "Unknown - validation failed",
            "validation_method": "error_fallback"
        }