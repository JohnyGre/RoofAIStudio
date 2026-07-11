# CHANGELOG

## Version 0.1.0 - 2023-10-26

### Initial Architecture Release

This release establishes the foundational architecture for Roof AI Studio, focusing on modularity, clean architecture principles, and extensibility. No full features are implemented, but the core structure for future development is in place.

**Key Architectural Components:**

*   **Core Application Framework:**
    *   `app_info.py`: Application metadata (name, version, author).
    *   `app_paths.py`: Centralized management of application directories.
    *   `config.py`: Centralized configuration management.
    *   `constants.py`: Application-wide constants.
    *   `logger.py`: Professional logging setup with rotating file and console output.
*   **Database Layer (SQLite with SQLAlchemy 2.x ORM):**
    *   Modular ORM models for: `Project`, `Customer`, `Roof`, `RoofPhoto`, `RoofGeometry`, `RoofPlane`, `RoofEdge`, `RoofVertex`, `Material`, `MaterialCategory`, `MaterialManufacturer`, `Estimate`, `EstimateItem`, `RoofTemplate`, `RoofTemplatePlane`, `AIModel`, `AITrainingSample`, `AIPrediction`, `ApplicationSettings`, `Supplier`, `PriceList`, `PriceItem`, `PriceHistory`, `LaborPrice`.
    *   `base.py`: Base classes for ORM models (UUID, timestamps).
    *   `enums.py`: Application-specific enums.
    *   `database.py`: SQLAlchemy engine and session management.
    *   `session.py`: Database session utility.
*   **User Interface (PySide6):**
    *   `main_window.py`: Main application window with menu bar, tool bar, and status bar.
    *   `workspace.py`: Central workspace layout with placeholder panels.
    *   `roof_canvas.py`: Interactive canvas for image display, zoom, pan, and geometry drawing (points, lines, polygons).
    *   `styles.py`: Centralized QSS styling for a dark theme.
*   **Image Processing Core (OpenCV, Pillow, NumPy):**
    *   `image_model.py`: Data models for image information.
    *   `image_loader.py`: Service for loading and validating image files, extracting metadata.
    *   `image_processor.py`: Service for common image manipulation (resize, rotate, crop).
*   **Geometry Engine (NumPy, Shapely):**
    *   Data models for `Point2D`, `Point3D`, `Edge`, `Polygon2D`, `RoofPlane`, `RoofGeometry`, `RoofGenome`.
    *   `calibration.py`: Services for image calibration (pixel to real-world units).
    *   `transform.py`: Services for geometric transformations.
    *   `measurement.py`: Services for calculating real-world measurements from geometry.
*   **Material Knowledge System:**
    *   Domain models for `Material`, `MaterialCategory`, `MaterialManufacturer`, `RoofSystem`, `RoofLayer`.
    *   `material_repository.py`: Abstract repository interface and SQLAlchemy implementation for material data access.
    *   `material_service.py`: Business logic for material quantity and cost calculations.
    *   `material_calculator.py`: Service to calculate required materials from roof measurements.
    *   `calculation_result.py`: Data model for material calculation output.
*   **AI Engine:**
    *   Abstract `AIModel` interface and `AIResult` data structures.
    *   `model_registry.py`: Singleton registry for managing AI models.
    *   `pipeline.py`: Core AI processing stages (preprocessing, inference, postprocessing).
    *   `ai_engine.py`: Orchestrates AI model management and prediction.
    *   `geometry_converter.py`: Converts AI prediction results into `RoofGeometry`.
    *   `prediction_pipeline.py`: High-level pipeline for roof analysis using AI.
    *   `models/roof_detector.py`: Placeholder OpenCV-based roof detector.
*   **Pricing Engine:**
    *   Data models for `LaborRate`, `PriceRule`, `EstimateLine`, `Estimate`.
    *   `pricing_service.py`: Business logic for cost calculations and rule application.
    *   `estimate_builder.py`: Service to build comprehensive customer estimates.
*   **PDF Export System (ReportLab):**
    *   Data models for `CompanyInfo`, `CustomerReport`.
    *   `report_template.py`: Defines layout and styles for PDF reports.
    *   `pdf_exporter.py`: Service for generating PDF reports and offers.
*   **Controllers:**
    *   `image_controller.py`: Mediates between UI and image loading/processing.
    *   `geometry_controller.py`: Mediates between UI and geometry editing/calculations.
*   **Integration Tests:**
    *   Comprehensive `pytest` suite covering `geometry`, `calibration`, `materials`, `pricing`, and the full `pipeline`.

**Installation:**

*   Updated `requirements.txt` with all project dependencies.

**Documentation:**

*   Updated `README.md` with project purpose, architecture overview, installation, running instructions, and a development roadmap.
