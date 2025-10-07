#!/usr/bin/env python3
"""
Pipeline Step Interfaces
Defines standardized input/output contracts for all pipeline steps
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
import numpy as np

@dataclass
class StepInput:
    """Standardized input for all pipeline steps"""
    # Core data
    data: Dict[str, Any] = field(default_factory=dict)

    # Image data
    image: Optional[np.ndarray] = None
    image_path: Optional[str] = None
    image_name: Optional[str] = None

    # Directory paths
    output_dir: Optional[Path] = None
    current_image_dir: Optional[Path] = None

    # Step-specific data from previous steps
    previous_results: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class StepOutput:
    """Standardized output for all pipeline steps"""
    success: bool
    step_name: str
    data: Dict[str, Any] = field(default_factory=dict)

    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # File outputs
    output_files: Dict[str, str] = field(default_factory=dict)

    # Metrics and timing
    processing_time: Optional[float] = None

    # For debugging and visualization
    debug_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineConfig:
    """Configuration for the entire pipeline"""
    output_dir: Path
    enable_visualization: bool = True
    enable_html_report: bool = True
    consensus_models: List[str] = field(default_factory=lambda: [
        'qwen2.5vl:7b',
        'minicpm-v:latest',
        'llava-llama3:latest'
    ])
    background_removal_enabled: bool = True
    debug_mode: bool = False

# Step-specific interfaces for type safety and clarity

@dataclass
class UIAnalysisResult:
    """Result from Step 1: UI Analysis"""
    header_region: Dict[str, int]
    content_region: Dict[str, int]
    footer_region: Dict[str, int]
    ui_structure: Dict[str, Any]
    compatibility_check: Dict[str, Any]

@dataclass
class CategoryAnalysisResult:
    """Result from Step 2: Category Analysis"""
    category_data: Dict[str, Any]
    detected_categories: List[str]
    active_category: Optional[str]
    header_text_data: Dict[str, Any]

@dataclass
class CanvasDetectionResult:
    """Result from Step 3: Canvas Detection"""
    canvases: List[Dict[str, Any]]
    product_grid_info: Dict[str, Any]
    canvas_coordinates: List[Dict[str, int]]

@dataclass
class ComponentExtractionResult:
    """Result from Step 4: Component Extraction"""
    components_data: List[Dict[str, Any]]
    coordinates: List[Dict[str, int]]
    clean_products: List[Dict[str, Any]]

@dataclass
class ConsensusAnalysisResult:
    """Result from Step 5: Consensus Analysis"""
    analyzed_products: List[Dict[str, Any]]
    consensus_results: List[Dict[str, Any]]
    confidence_scores: List[float]

@dataclass
class CSVGenerationResult:
    """Result from Step 6: CSV Generation"""
    csv_data: List[Dict[str, Any]]
    csv_path: str
    total_products: int
    successful_analyses: int
    html_report_path: Optional[str] = None