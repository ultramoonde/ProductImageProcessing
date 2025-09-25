# ProductImageProcessing Pipeline

A sophisticated food product image extraction pipeline designed to process grocery app screenshots (specifically Flink app) and extract clean product images with metadata.

## 🌟 Features

- **6-Step Pipeline Architecture** - Systematic processing from UI analysis to final product extraction
- **3-Model LLM Consensus System** - Uses llama3.2-vision:11b, minicpm-v:latest, moondream:latest for reliable analysis
- **UI Compatibility Validation** - Ensures screenshots match expected Flink app format (530px header requirement)
- **Advanced Pink Button Removal** - HSV color detection + HoughCircles method for clean product extraction
- **Background Removal** - Produces clean, isolated product images suitable for e-commerce
- **Comprehensive Reporting** - CSV generation and HTML visualization of the entire pipeline

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama and download required models
ollama serve &
ollama pull llama3.2-vision:11b
ollama pull minicpm-v:latest
ollama pull moondream:latest
```

### Basic Usage

```bash
# Process a single image
python step_by_step_pipeline.py --image path/to/flink_screenshot.png

# Test with compatibility check
python step_by_step_pipeline.py --test path/to/flink_screenshot.png
```

### Python API

```python
from step_by_step_pipeline import StepByStepPipeline

pipeline = StepByStepPipeline()
results = pipeline.process_image("path/to/flink_screenshot.png")

if results.get("success"):
    print(f"✅ Extracted {len(results['products'])} products")
    print(f"📁 Results in: {results['output_dir']}")
```

## 📋 Pipeline Steps

1. **Step 0**: UI Analysis & Compatibility Check
2. **Step 1**: Header/Content Region Detection
3. **Step 2**: Product Grid Canvas Detection
4. **Step 3**: Individual Component Extraction
5. **Step 4**: Product Tile Processing & Pink Button Removal
6. **Step 5**: Background Removal & Final Export

## 🔧 Core Components

### Main Pipeline
- `step_by_step_pipeline.py` - Main orchestration and processing logic

### Source Modules (`src/`)
- `local_consensus_analyzer.py` - 3-model LLM consensus system
- `image_processor.py` - Image processing utilities
- `text_extractor.py` - OCR and text analysis
- `vision_analyzer.py` - Computer vision components
- `background_removal_*.py` - Background removal services

## 📊 Expected Input & Output

### Input Requirements
- **Format**: PNG screenshots from Flink grocery app
- **Header Height**: Exactly 530 pixels
- **Content Start**: Pixel row 531
- **Layout**: Standard Flink product grid below header

### Output Structure
```
step_by_step_flat/
├── {image}_01_annotated.jpg         # UI regions visualization
├── {image}_02_canvases.jpg          # Product grids detected
├── {image}_03_components.jpg        # Individual products identified
├── {image}_component_N_clean_product.png  # Clean product images
├── {image}_component_N_text_region.png   # Associated text regions
├── {image}_pipeline_report.html     # Complete processing report
└── FINAL_EXTRACTED_PRODUCTS.csv    # Product metadata
```

## 🛡️ Configuration

### Environment Variables
```bash
export ANTHROPIC_API_KEY="your_claude_key"  # For vision analysis (optional)
```

### Ollama Models
The pipeline uses local Ollama models by default:
- `llama3.2-vision:11b` - Primary vision analysis
- `minicpm-v:latest` - Secondary vision analysis
- `moondream:latest` - Tertiary vision analysis

## 📚 Documentation

- `HANDOVER_DOCUMENTATION.md` - Complete technical handover guide
- `PIPELINE_STEPS_DETAILED.md` - Detailed step-by-step documentation
- `examples/` - Usage examples and tutorials

## ⚡ Performance

- **Processing Time**: ~30-60 seconds per screenshot (6 steps)
- **Consensus Accuracy**: 3-model agreement for reliable results
- **Output Quality**: Clean product images suitable for e-commerce catalogs

## 🔍 Troubleshooting

### Common Issues

**"Incompatible header height"**
- Ensure screenshot has exactly 530px header height
- Use screenshots directly from Flink grocery app

**"No products found"**
- Verify image contains product grid/tiles
- Check image quality and resolution

**"Ollama connection failed"**
- Ensure Ollama server is running: `ollama serve`
- Verify models are downloaded: `ollama list`

## 📄 License

This project is part of the UltraMoon development ecosystem.

## 🤝 Contributing

This pipeline was developed with sophisticated computer vision and AI techniques. For technical details and contribution guidelines, see the handover documentation.

---

*Generated with Claude Code - Anthropic's official CLI for Claude*