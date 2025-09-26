#!/usr/bin/env python3
"""
Step 6: CSV Aggregation (Fixed)
Aggregates all pipeline results into a comprehensive final CSV file
"""

import sys
import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project paths for imports
sys.path.append('src')
sys.path.append('.')

from src.interfaces import StepInput, StepOutput

def run(input_data: StepInput) -> StepOutput:
    """
    Aggregate all pipeline results into a comprehensive final CSV

    Args:
        input_data: StepInput with all previous step data

    Returns:
        StepOutput with CSV aggregation results
    """

    print("\n📊 STEP 6: Final CSV Aggregation")
    print("=" * 50)

    output_dir = input_data.current_image_dir
    image_name = input_data.image_name

    if not output_dir:
        return StepOutput(
            success=False,
            step_name="CSV Aggregation",
            data={"error": "Output directory not specified"}
        )

    try:
        # Initialize CSV aggregator
        aggregator = CSVAggregator(output_dir, image_name)

        # Aggregate all data sources
        final_csv_data = aggregator.aggregate_all_data()

        # Generate comprehensive CSV
        csv_path = aggregator.generate_final_csv(final_csv_data)

        print(f"📁 Final CSV saved to: {csv_path}")
        print(f"📊 Total rows: {len(final_csv_data)}")

        return StepOutput(
            success=True,
            step_name="CSV Aggregation",
            data={
                "csv_path": str(csv_path),
                "total_products": len(final_csv_data),
                "aggregated_sources": aggregator.get_data_sources_used()
            },
            output_files={"final_csv": str(csv_path)}
        )

    except Exception as e:
        print(f"❌ CSV Aggregation failed: {str(e)}")
        return StepOutput(
            success=False,
            step_name="CSV Aggregation",
            data={"error": str(e)}
        )


class CSVAggregator:
    """Aggregates pipeline data from multiple sources into final CSV"""

    def __init__(self, output_dir: Path, image_name: str):
        self.output_dir = Path(output_dir)
        self.image_name = image_name
        self.data_sources_used = []

    def aggregate_all_data(self) -> List[Dict]:
        """Aggregate data from all available sources"""
        print("🔄 Aggregating data from all pipeline steps...")

        # Step 1: Load category context
        category_data = self._load_category_data()

        # Step 2: Load component coordinates
        components_data = self._load_components_data()

        # Step 3: Load consensus analysis (main product data)
        products_data = self._load_consensus_data()

        # Step 4: Merge all data sources
        final_data = self._merge_all_sources(category_data, components_data, products_data)

        print(f"✅ Successfully aggregated {len(final_data)} product records")
        return final_data

    def _load_category_data(self) -> Dict:
        """Load category analysis results"""
        category_file = self.output_dir / f"{self.image_name}_02_analysis.json"

        if category_file.exists():
            with open(category_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.data_sources_used.append("category_analysis")
                print(f"   📂 Loaded category data: {data.get('main_category', 'Unknown')}")
                return data
        else:
            print("   ⚠️ Category analysis file not found")
            return {}

    def _load_components_data(self) -> List[Dict]:
        """Load component coordinates from CSV"""
        components_file = self.output_dir / f"{self.image_name}_03_components.csv"

        if components_file.exists():
            components = []
            with open(components_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                components = list(reader)

            self.data_sources_used.append("components_coordinates")
            print(f"   📂 Loaded {len(components)} component coordinate records")
            return components
        else:
            print("   ⚠️ Components CSV file not found")
            return []

    def _load_consensus_data(self) -> Dict:
        """Load consensus analysis results"""
        consensus_file = self.output_dir / f"{self.image_name}_05_consensus_analysis.json"

        if consensus_file.exists():
            with open(consensus_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.data_sources_used.append("consensus_analysis")
                analyzed_products = data.get('analyzed_products', [])
                print(f"   📂 Loaded {len(analyzed_products)} consensus product records")
                return data
        else:
            print("   ⚠️ Consensus analysis file not found")
            return {}

    def _merge_all_sources(self, category_data: Dict, components_data: List[Dict], products_data: Dict) -> List[Dict]:
        """Merge all data sources into comprehensive product records"""

        merged_products = []
        analyzed_products = products_data.get('analyzed_products', [])
        category_context = products_data.get('category_context', category_data)

        print(f"🔄 Merging data from {len(analyzed_products)} products...")

        for i, product in enumerate(analyzed_products):
            # Start with product data as base
            merged_product = {
                # Basic product information
                "product_id": product.get("product_id", f"{self.image_name}_product_{i+1}"),
                "product_number": product.get("product_number", i+1),
                "product_name": product.get("product_name", ""),
                "brand": product.get("brand", ""),
                "price": product.get("price", ""),
                "original_price": product.get("original_price", ""),
                "unit": product.get("unit", ""),
                "weight_quantity": product.get("weight", ""),
                "price_per_kg": product.get("price_per_kg", ""),
                "price_per_piece": product.get("price_per_piece", ""),
                "price_per_liter": product.get("price_per_liter", ""),
                "description": product.get("description", ""),

                # Category information (use corrected categories from validation)
                "main_category": category_context.get("main_category", ""),
                "active_subcategory": category_context.get("active_subcategory", ""),
                "category_confidence": category_context.get("confidence", 0.0),
                "category_method": category_context.get("method", ""),

                # Processing metadata
                "consensus_confidence": product.get("consensus_confidence", 0.0),
                "models_agreed": product.get("models_agreed", 0),
                "total_models": product.get("total_models", 3),

                # File paths
                "clean_image_path": product.get("clean_image_path", ""),
                "text_image_path": product.get("text_image_path", ""),
                "hole_punched_image_path": "",  # Will be filled from components
                "background_removed_image_path": "",  # Will be filled from components

                # Source image info
                "source_image": self.image_name,
                "processing_timestamp": datetime.now().isoformat(),
            }

            # Find matching component data
            component_match = self._find_matching_component(product, components_data)
            if component_match:
                merged_product.update({
                    # Canvas/Component coordinates
                    "canvas_x": int(component_match.get("canvas_x", 0)),
                    "canvas_y": int(component_match.get("canvas_y", 0)),
                    "canvas_width": int(component_match.get("canvas_width", 0)),
                    "canvas_height": int(component_match.get("canvas_height", 0)),

                    # Product tile coordinates
                    "product_image_x": int(component_match.get("product_image_x", 0)),
                    "product_image_y": int(component_match.get("product_image_y", 0)),
                    "product_image_width": int(component_match.get("product_image_width", 0)),
                    "product_image_height": int(component_match.get("product_image_height", 0)),

                    # Pink button detection
                    "pink_button_detected": component_match.get("pink_button_detected", "False").lower() == "true",
                    "pink_button_x": int(component_match.get("pink_button_x", 0)) if component_match.get("pink_button_x") else None,
                    "pink_button_y": int(component_match.get("pink_button_y", 0)) if component_match.get("pink_button_y") else None,
                    "pink_button_center_x": int(component_match.get("pink_button_center_x", 0)) if component_match.get("pink_button_center_x") else None,
                    "pink_button_center_y": int(component_match.get("pink_button_center_y", 0)) if component_match.get("pink_button_center_y") else None,
                    "pink_button_confidence": float(component_match.get("pink_button_confidence", 0.0)),

                    # Derive image file paths
                    "hole_punched_image_path": f"{self.image_name}_component_{i+1}_product_holes.png",
                    "background_removed_image_path": f"{self.image_name}_component_{i+1}_product_holes.png_product_nobg.png"
                })

            # Add validation information if available
            validation_info = category_context.get("validation_info", {})
            if validation_info:
                merged_product.update({
                    "category_validation_performed": validation_info.get("validated", False),
                    "category_override_confidence": validation_info.get("product_match_confidence", 0.0),
                    "category_validation_reasoning": validation_info.get("reasoning", "")
                })

            merged_products.append(merged_product)

        return merged_products

    def _find_matching_component(self, product: Dict, components_data: List[Dict]) -> Optional[Dict]:
        """Find matching component data for a product"""
        product_number = product.get("product_number")

        if product_number:
            # Match by component number
            for component in components_data:
                component_id = component.get("component_id", "")
                if f"component_{product_number}" in component_id:
                    return component

        return None

    def generate_final_csv(self, data: List[Dict]) -> Path:
        """Generate the final comprehensive CSV file"""
        csv_path = self.output_dir / f"{self.image_name}_FINAL_RESULTS.csv"

        if not data:
            # Create empty CSV with headers
            headers = [
                "product_id", "product_number", "product_name", "brand", "price", "original_price",
                "unit", "weight_quantity", "price_per_kg", "price_per_piece", "price_per_liter",
                "description", "main_category", "active_subcategory", "category_confidence",
                "category_method", "consensus_confidence", "models_agreed", "total_models",
                "source_image", "processing_timestamp"
            ]

            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

            return csv_path

        # Use pandas for better CSV handling
        df = pd.DataFrame(data)

        # Reorder columns for better readability
        column_order = [
            # Product identification
            "product_id", "product_number", "product_name", "brand",

            # Pricing & quantity
            "price", "original_price", "unit", "weight_quantity",
            "price_per_kg", "price_per_piece", "price_per_liter",

            # Category classification
            "main_category", "active_subcategory", "category_confidence", "category_method",

            # Coordinates & detection
            "canvas_x", "canvas_y", "canvas_width", "canvas_height",
            "product_image_x", "product_image_y", "product_image_width", "product_image_height",
            "pink_button_detected", "pink_button_center_x", "pink_button_center_y", "pink_button_confidence",

            # Quality metrics
            "consensus_confidence", "models_agreed", "total_models",

            # File references
            "clean_image_path", "hole_punched_image_path", "background_removed_image_path",

            # Metadata
            "source_image", "processing_timestamp", "description"
        ]

        # Add validation columns if they exist
        if "category_validation_performed" in df.columns:
            column_order.extend(["category_validation_performed", "category_override_confidence", "category_validation_reasoning"])

        # Reorder columns (keep only existing ones)
        existing_columns = [col for col in column_order if col in df.columns]
        remaining_columns = [col for col in df.columns if col not in existing_columns]
        final_column_order = existing_columns + remaining_columns

        df = df[final_column_order]

        # Save to CSV with good formatting
        df.to_csv(csv_path, index=False, encoding='utf-8')

        return csv_path

    def get_data_sources_used(self) -> List[str]:
        """Return list of data sources successfully loaded"""
        return self.data_sources_used