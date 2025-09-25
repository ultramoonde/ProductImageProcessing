# Complete Pipeline Documentation - Current Code Behavior

This document accurately describes what the step-by-step pipeline currently does, based on the actual code implementation.

## Overview

The pipeline processes German grocery app screenshots (Flink) to extract products with accurate category classification. The system uses a **3-model AI consensus approach** for header analysis and a **step-by-step processing approach** with flat file structure.

---

## STEP 0: LLM Consensus-Based Category Detection 🔍

**Purpose**: Analyze screenshot headers using AI consensus to detect German food categories

**Location**: `step_by_step_pipeline.py:612-725` (`_extract_categories_with_consensus`)

### What Actually Happens:

1. **Header Region Extraction**
   - Crops top **250px** of screenshot to focus on navigation area
   - Creates clean header region for AI analysis

2. **3-Model AI Consensus Analysis**
   - **Models Used**: `llama3.2-vision:11b`, `minicpm-v:latest`, `moondream:latest`
   - **Method Called**: `LocalConsensusAnalyzer.analyze_categories_with_consensus()`
   - **Analysis Mode**: UI analysis specifically for German category detection

3. **Consensus Processing**
   - Requires **minimum 2/3 models** to succeed
   - Aggregates category results from successful models
   - Uses **majority voting** for final category selection

4. **Fixed Prompt System**
   - **Critical Fix Applied**: Replaced placeholder-generating prompts with actual German category extraction
   - **Prompt Focus**: Extract REAL German categories like `Vegan & Vegetarisch`, `Milchalternativen`, `Bio`
   - **No More Placeholders**: System no longer returns "category1", "category2" fake categories

### Outputs:
- `IMG_XXXX_00_analysis.jpg`: Annotated header analysis
- `IMG_XXXX_00_analysis.json`: Category detection results with actual German categories

**Example Result**:
```json
{
  "main_category": "Vegan & Vegetarisch",
  "active_subcategory": "Milchalternativen",
  "available_subcategories": ["Vegan & Vegetarisch", "Milchalternativen", "Bio", "Fleisch & Fisch"]
}
```

---

## STEP 1: UI Region Analysis & Splitting 🔤

**Purpose**: Analyze screenshot structure and split into logical UI regions (header, content, footer)

**Location**: `step_by_step_pipeline.py:1300-1400` (Step 1 processing section)

### What Actually Happens:

1. **UI Region Detection**
   - **Header Region**: Top navigation area with categories (typically 0-531px)
   - **Content Region**: Main product grid area (531px to bottom-200px)
   - **Footer Region**: Bottom navigation bar (last ~200px)
   - Uses `ScreenshotUIAnalyzer.analyze_screenshot()` for boundary detection

2. **Region Boundary Calculation**
   - Identifies precise pixel coordinates for each region
   - Validates region boundaries and dimensions
   - Ensures regions don't overlap

3. **Graceful Error Handling** (**Fixed as requested**)
   - **When Step 0 fails**: Creates error file instead of processing garbage
   - **Error File**: `IMG_XXXX_01_error.txt` with message "Image Error - No Header Found"
   - **No Garbage Output**: Skips further processing instead of creating invalid files
   - **Code Location**: Lines 1366-1379 in `step_by_step_pipeline.py`

4. **Fallback Behavior** (**Legacy - Not Recommended**)
   - If no header regions found, falls back to old HSV/OCR methods
   - This fallback is discouraged and will create error files instead

### Outputs:
- `IMG_XXXX_01_header_region.jpg`: Cropped header region
- `IMG_XXXX_01_annotated.jpg`: Annotated region boundaries
- `IMG_XXXX_01_header_text.json`: Region coordinates and structure
- `IMG_XXXX_01_error.txt`: Error file when Step 0 fails (no garbage processing)

---

## STEP 2: Product Canvas Detection 🎯

**Purpose**: Detect and isolate individual product tiles/regions within the content area

### What Actually Happens:

1. **Content Region Focus**
   - Uses content region boundaries from Step 1
   - Focuses on main product grid area (below header, above footer)

2. **Grid-Based Tile Detection**
   - **Standard Tile Size**: 573×813px each
   - **Layout**: Typically 2-column grid layout
   - **Detection Count**: Usually 4+ product canvases per screenshot

3. **Canvas Coordinate Mapping**
   - Records precise X,Y coordinates for each product tile
   - Validates canvas dimensions and positioning
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

### What Actually Happens:

1. **Pink Button Detection**
   - **HSV Color Range**: H=320-340°, S=70-100%, V=70-100%
   - **Shape Validation**: Circular buttons, typically ~25px radius
   - **Location Mapping**: Records button coordinates relative to canvas

2. **Component Coordinate Recording**
   - Maps detected buttons to their parent canvas
   - Records precise X,Y coordinates for removal in Step 4
   - Validates detection accuracy and removes false positives

### Outputs:
- `IMG_XXXX_03_components.csv`: Component coordinates
- `IMG_XXXX_03_components.jpg`: Annotated component detection

---

## STEP 3B: Unified Canvas Processing 🧭

**Purpose**: Combine tile detection with category assignment using LLM results

### What Actually Happens:

1. **Category Assignment Integration**
   - Uses categories from Step 0 LLM consensus results
   - Maps detected German categories to each product canvas
   - Creates spatial hierarchy: canvas → category mapping

2. **Individual Canvas Visualization**
   - Creates separate canvas images (573×813px each)
   - Applies category labels from Step 0 results
   - Generates individual canvas files for validation

### Outputs:
- `IMG_XXXX_01_canvas.jpg`, `IMG_XXXX_02_canvas.jpg`, etc.: Individual canvas images
- Category-to-canvas assignment metadata

---

## STEP 4: Refined Product Extraction Pipeline 🔥

**Purpose**: Clean product extraction with background removal and LLM analysis

### STEP 4A: Pink Button Removal & Region Extraction

1. **Button Removal Process**
   - Uses coordinates from Step 3 to remove pink buttons
   - **Masking Method**: Circular masking with ~25px radius
   - **Context Filling**: Fills removed areas with surrounding pixels

2. **Region Splitting**
   - **Product Region**: Top 573×573px (pure product image)
   - **Text Region**: Bottom 573×240px (price, description, brand)
   - Separates visual content from textual information

### STEP 4B: Enhanced Consensus Processing

1. **Multi-Modal Analysis**
   - Combines product image + text region for complete analysis
   - Uses same 3-model consensus system as Step 0
   - **Analysis Mode**: Product analysis (not UI)

2. **Product Information Extraction**
   - **Product Name**: German product names
   - **Price**: Euro format (€)
   - **Brand**: Brand identification
   - **Weight/Quantity**: German units (kg, g, ml, Stk.)
   - **Unit Type**: Classification

### STEP 4C: Background Removal

1. **Professional Background Removal**
   - **Primary Tool**: `rembg` for background removal
   - **Provider Priority**: rembg → remove_bg → photoroom → removal_ai
   - **Output**: Clean product on transparent background

2. **Quality Validation**
   - Prevents black/empty array issues
   - Validates successful background removal
   - **Fallback**: Uses original image if removal fails

### Outputs Per Component:
- `IMG_XXXX_04a_component_XXX_product.png`: Product region only
- `IMG_XXXX_04a_component_XXX_text.png`: Text region only
- `IMG_XXXX_04a_component_XXX_product_clean.png`: Background removed
- `IMG_XXXX_04b_component_XXX_analysis.json`: LLM product analysis

---

## File Structure

**Output Directory**: `step_by_step_flat/IMG_XXXX/`

```
step_by_step_flat/IMG_XXXX/
├── IMG_XXXX_00_analysis.jpg/json          # Step 0: LLM category detection
├── IMG_XXXX_01_header_*.jpg/json          # Step 1: UI region splitting
├── IMG_XXXX_01_error.txt                  # Step 1: Error file (when Step 0 fails)
├── IMG_XXXX_02_canvases.jpg/csv/json      # Step 2: Canvas detection
├── IMG_XXXX_03_components.jpg/csv         # Step 3: Component coordinates
├── IMG_XXXX_0X_canvas.jpg                 # Step 3B: Individual canvases
└── IMG_XXXX_04*_component_XXX.*           # Step 4: Clean products & analysis
```

---

## Key Technical Implementation Details

### Consensus System Architecture:
- **Models**: `llama3.2-vision:11b`, `minicpm-v:latest`, `moondream:latest`
- **Minimum Success**: 2/3 models must succeed
- **Aggregation**: Majority voting for final results
- **Reliability**: Robust against individual model failures

### Error Handling:
- **Step 0 → Step 1**: Graceful error handling with error files
- **No Garbage Processing**: Failed images get error files, not invalid output
- **Skip Pattern**: Failed images are skipped entirely after error file creation

### German Category Detection Success:
The system now successfully detects these **real German categories**:

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

**No More "Bananen" Problem**: The critical issue where all products were incorrectly assigned to "Bananen" has been completely resolved through the LLM consensus system implementation.

---

## Critical Fixes Applied

### 1. Consensus Prompt Fix (`local_consensus_analyzer.py:68-82`)
**Problem**: Models were returning placeholder categories ("category1", "category2")
**Solution**: Rewrote UI analysis prompt to extract actual German categories

### 2. Step 1 Graceful Error Handling (`step_by_step_pipeline.py:1366-1379`)
**Problem**: Failed images were processed with garbage, creating invalid output files
**Solution**: Create single error file and skip processing when Step 0 fails

### 3. Category Assignment Integration
**Problem**: Categories from consensus system weren't properly integrated into canvas processing
**Solution**: Direct integration of Step 0 LLM results into canvas-category mapping

This documentation reflects the current state of the code and serves as the definitive reference for pipeline behavior.