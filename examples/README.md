# Examples

This directory contains usage examples for the ProductImageProcessing pipeline.

## Basic Usage

```python
python examples/basic_usage.py
```

Basic example showing how to process a single Flink grocery app screenshot.

## Requirements

- A valid Flink grocery app screenshot (PNG format)
- All dependencies installed: `pip install -r requirements.txt`
- Ollama running with required models:
  ```bash
  ollama serve &
  ollama pull llama3.2-vision:11b
  ollama pull minicpm-v:latest
  ollama pull moondream:latest
  ```

## Expected Input

The pipeline expects Flink grocery app screenshots with:
- Exactly 530px header height
- Product grid/tiles below the header
- Standard Flink UI layout

## Output

The pipeline generates:
- Step-by-step visualization images
- Clean product images (background removed)
- CSV file with product metadata
- HTML report showing the complete pipeline

## Example Structure

```
step_by_step_flat/
├── IMG_8104_01_annotated.jpg         # UI regions identified
├── IMG_8104_02_canvases.jpg          # Product grids detected
├── IMG_8104_03_components.jpg        # Individual products found
├── IMG_8104_component_1_clean_product.png  # Clean product images
├── IMG_8104_component_1_text_region.png   # Text regions
└── IMG_8104_pipeline_report.html     # Complete report
```