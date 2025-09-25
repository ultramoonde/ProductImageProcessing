# Step-by-Step Pipeline Analysis - Complete Breakdown

This document describes the complete step-by-step pipeline with the new **LLM consensus-based category detection system** that replaced the old HSV/OCR approach.

## Overview

The pipeline processes German grocery app screenshots (Flink) to extract products with accurate category classification. The system now uses a **3-model AI consensus approach** for robust analysis.

---

## STEP 0: UI Analysis & Category Detection 🔍

**Purpose**: Analyze the screenshot header to detect available German food categories

### What Happens:
1. **Header Region Extraction**
   - Crops top 250px of screenshot to focus on navigation area
   - Removes status bar and focuses on category tabs

2. **LLM Consensus Analysis**
   - **Models Used**: `llama3.2-vision:11b`, `minicpm-v:latest`, `moondream:latest`
   - **Analysis Mode**: UI analysis specifically for German category detection
   - **Prompt**: Instructs models to find actual German category names (not placeholders)

3. **Category Detection**
   - Detects real categories like: `Vegan & Vegetarisch`, `Milchalternativen`, `Fleisch & Fisch`, `Bio`, `Tiefkühl`
   - Identifies currently active/highlighted category if visible
   - No more hardcoded HSV color detection or EasyOCR complexity

### Outputs:
- `IMG_XXXX_00_analysis.jpg`: Annotated header analysis
- `IMG_XXXX_00_analysis.json`: Category detection results
- **Categories List**: Real German food categories for use in subsequent steps

**Example Result**:
```json
{
  "main_category": "Vegan & Vegetarisch",
  "active_subcategory": "Milchalternativen",
  "available_subcategories": ["Vegan & Vegetarisch", "Milchalternativen", "Bio", "Fleisch & Fisch", "Getränke"]
}
```

---

## STEP 1: UI Region Analysis & Splitting 🔤

**Purpose**: Analyze screenshot structure and split into logical UI regions (header, content, footer)

### What Happens:
1. **UI Region Detection** (`ScreenshotUIAnalyzer.analyze_screenshot()`)
   - **Header Region**: Top navigation area with categories (usually ~0-531px)
   - **Content Region**: Main product grid area (usually ~531px to bottom-200px)
   - **Footer Region**: Bottom navigation bar (usually last ~200px)
   - Uses visual break analysis and heuristics

2. **Region Boundary Detection**
   - Identifies precise pixel coordinates for each region
   - Calculates region dimensions and positions
   - Validates region boundaries make sense

3. **Header Category Analysis** (when needed)
   - If no categories from Step 0, extracts from header using old HSV/OCR methods
   - This is fallback only - Step 0 LLM consensus is primary
   - Only happens when `print("🔍 No header region found, performing UI analysis...")`

4. **Basic Region Validation**
   - Ensures regions don't overlap
   - Validates minimum/maximum region sizes
   - **NOTE**: Product tile detection is disabled in Step 1 (see line 51-58)

### Outputs:
- `IMG_XXXX_01_header_region.jpg`: Cropped header region
- `IMG_XXXX_01_annotated.jpg`: Annotated region boundaries
- `IMG_XXXX_01_header_text.json`: Region coordinates and structure

### Why This Changed:
This step **didn't change** - I was wrong in my documentation. The UI splitting was always Step 1. Only Step 0 changed from HSV/OCR to LLM consensus.

---

## STEP 2: Product Canvas Detection 🎯

**Purpose**: Detect and isolate individual product tiles/regions

### What Happens:
1. **Content Region Analysis**
   - Identifies main content area below header (starts ~Y=531px)
   - Calculates available space for product grid

2. **Tile Detection**
   - **Standard Tile Size**: 573×813px each
   - Detects grid layout (typically 2 columns)
   - Identifies 4+ product canvases per screenshot

3. **Canvas Validation**
   - Validates each detected region contains product content
   - Ensures proper spacing and alignment
   - Removes invalid or overlapping regions

### Outputs:
- `IMG_XXXX_02_product_canvases.csv`: Canvas coordinates and metadata
- `IMG_XXXX_02_canvases.jpg`: Visualized product grid
- `IMG_XXXX_02_canvases.json`: Canvas detection results

**Canvas Structure**:
```json
{
  "canvas_id": 1,
  "x": 48, "y": 673,
  "width": 573, "height": 813,
  "main_category": "Vegan & Vegetarisch",
  "subcategory": "Milchalternativen"
}
```

---

## STEP 3: Component Coordinate Extraction 📍

**Purpose**: Detect UI elements within each product canvas (pink buttons, etc.)

### What Happens:
1. **Pink Button Detection**
   - Searches for pink circular "add to cart" buttons
   - Uses HSV color detection: `H=320-340°, S=70-100%, V=70-100%`
   - Validates button shape and size (typically ~25px radius)

2. **Component Mapping**
   - Maps detected buttons to their parent canvas
   - Records precise coordinates for UI element removal
   - Validates detection accuracy

3. **Coordinate Validation**
   - Ensures buttons are within expected regions
   - Removes false positives
   - Prepares coordinates for Step 4 processing

### Outputs:
- `IMG_XXXX_03_components.csv`: Component coordinates
- `IMG_XXXX_03_components.jpg`: Annotated component detection

---

## STEP 3B: Unified Canvas Processing 🧭

**Purpose**: Combine tile detection with category assignment using LLM results

### What Happens:
1. **Category Assignment**
   - Uses categories from Step 0 LLM consensus results
   - Assigns detected categories to each product canvas
   - Creates category-to-canvas mapping

2. **Canvas Visualization**
   - Creates individual canvas images (573×813px)
   - Applies category labels to each canvas
   - Generates visualization for validation

### Outputs:
- Individual canvas files: `IMG_XXXX_01_canvas.jpg`, `IMG_XXXX_02_canvas.jpg`, etc.
- Category assignment metadata

---

## STEP 4: Refined Product Extraction Pipeline 🔥

**Purpose**: Clean product extraction with background removal and LLM analysis

### STEP 4A: Pink Button Removal & Region Extraction
1. **Button Removal**
   - Uses coordinates from Step 3 to remove pink buttons
   - Applies circular masking with ~25px radius
   - Fills removed areas with surrounding context

2. **Region Extraction**
   - **Product Region**: Top 573×573px (product image)
   - **Text Region**: Bottom 573×240px (price, description)
   - Separates visual and textual content

### STEP 4B: Enhanced Consensus Processing
1. **Multi-Modal Analysis**
   - Combines product image + text region
   - Uses 3-model LLM consensus system
   - **Mode**: Product analysis (not UI)

2. **Product Information Extraction**
   - Product name (German)
   - Price (€ format)
   - Brand name
   - Weight/quantity (kg, g, ml, Stk.)
   - Unit type

### STEP 4C: Background Removal
1. **Clean Product Isolation**
   - Uses `rembg` for background removal
   - **Provider Priority**: rembg → remove_bg → photoroom → removal_ai
   - Isolates product on transparent background

2. **Quality Validation**
   - Prevents black/empty array issues
   - Validates successful background removal
   - Fallback to original if removal fails

### Outputs Per Component:
- `IMG_XXXX_04a_component_XXX_product.png`: Product region only
- `IMG_XXXX_04a_component_XXX_text.png`: Text region only
- `IMG_XXXX_04a_component_XXX_product_clean.png`: Background removed
- `IMG_XXXX_04b_component_XXX_analysis.json`: LLM analysis results

---

## Key Improvements with LLM System

### Before (Old HSV/OCR System):
- ❌ Hardcoded HSV color detection
- ❌ Complex EasyOCR preprocessing
- ❌ Always assigned products to "Bananen" category
- ❌ Failed to detect actual German categories

### After (New LLM Consensus System):
- ✅ **AI-powered category detection** using 3 vision models
- ✅ **Real German categories**: `Vegan & Vegetarisch`, `Milchalternativen`, `Bio`, etc.
- ✅ **Consensus-based accuracy** - majority voting across models
- ✅ **Simplified pipeline** - header analysis directly to consensus
- ✅ **Robust detection** - handles various German grocery categories

---

## Technical Architecture

### Models Used:
1. **llama3.2-vision:11b** - Primary vision-language model
2. **minicpm-v:latest** - Compact multimodal model
3. **moondream:latest** - Specialized vision model

### Consensus Logic:
- **Minimum 2/3 models** must succeed for valid result
- **Category aggregation** combines results from all successful models
- **Confidence scoring** based on model agreement
- **Fallback handling** for model failures

### File Structure:
```
step_by_step_flat/IMG_XXXX/
├── IMG_XXXX_00_analysis.jpg/json          # Step 0: Category detection
├── IMG_XXXX_01_header_*.jpg/json          # Step 1: Header text
├── IMG_XXXX_02_canvases.jpg/csv/json      # Step 2: Canvas detection
├── IMG_XXXX_03_components.jpg/csv         # Step 3: Component coords
├── IMG_XXXX_0X_canvas.jpg                 # Step 3B: Individual canvases
└── IMG_XXXX_04*_component_XXX.*           # Step 4: Clean products
```

---

## Categories Successfully Detected

The new system accurately detects these German grocery categories:

**Main Categories:**
- `Obst` (Fruit)
- `Gemüse` (Vegetables)
- `Fleisch & Fisch` (Meat & Fish)
- `Vegan & Vegetarisch` (Vegan & Vegetarian)
- `Bio` (Organic)
- `Backwaren` (Bakery)
- `Getränke` (Beverages)
- `Tiefkühl` (Frozen)

**Subcategories:**
- `Milchalternativen` (Milk Alternatives)
- `Äpfel & Birnen` (Apples & Pears)
- `Joghurt & Desserts`
- `Nudeln` (Noodles/Pasta)
- `Bananen` (Bananas)
- `Brotaufstriche` (Spreads)

This represents a complete transformation from the previous system that incorrectly assigned everything to "Bananen" regardless of actual content.