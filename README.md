# Roof AI Studio

Roof AI Studio is a professional desktop application designed for analyzing and modeling roof structures, incorporating AI-powered features for enhanced efficiency and accuracy. It aims to streamline the process of roof estimation, measurement, and material calculation for construction professionals.

## Project Architecture Overview

This project follows a Clean Architecture and SOLID principles, ensuring a modular, maintainable, and scalable design. Key architectural decisions include:

*   **Modular Design:** The application is divided into distinct modules (e.g., `core`, `database`, `geometry`, `ai`, `materials`, `pricing`, `ui`, `exporters`) with clear responsibilities.
*   **Layered Structure:** A clear separation between UI, controllers, services, and core domain logic.
*   **Dependency Injection:** Services and controllers are designed to receive their dependencies, promoting testability and flexibility.
*   **Type Hinting:** Extensive use of Python type hints for improved code readability, maintainability, and error detection.
*   **Extensibility:** Designed to easily integrate new AI models and support a plugin architecture for future features.

### Folder Structure:

-   `app/`: Contains the core application logic.
    -   `ai/`: Modules related to Artificial Intelligence functionalities, such as object detection, segmentation, and geometry prediction. Includes abstract interfaces for AI models, result data structures, a model registry, and a processing pipeline.
    -   `calibration/`: Handles calibration processes for various inputs, including camera feeds or measurement tools, to ensure accurate data acquisition.
    -   `controllers/`: Manages the application's flow, acting as an intermediary between the UI and the business logic/services.
    -   `core/`: Core application-wide services and utilities, independent of specific frameworks or UI. Includes application info, path management, configuration, logging, and image processing.
    -   `database/`: Manages database interactions using SQLAlchemy ORM for SQLite. Contains base models, enums, and specific ORM models for various entities (projects, customers, roofs, materials, pricing, AI data, settings).
    -   `exporters/`: Modules responsible for exporting data in various formats, currently supporting professional PDF report generation using ReportLab.
    -   `geometry/`: Handles geometric calculations, transformations, and representations of roof elements. Includes 2D/3D points, edges, polygons, roof planes, and comprehensive roof geometry models.
    -   `materials/`: Manages material properties, catalogs, and related data for roof construction, including material models, repositories, and calculation services.
    -   `models/`: Defines the data structures and entities used throughout the application (currently integrated within other modules like `database` and `geometry`).
    -   `plugins/`: Provides an extensible architecture for adding new functionalities or integrations (placeholder for future development).
    -   `pricing/`: Contains logic for calculating costs, generating quotes, and managing pricing rules, including labor rates, price rules, and estimate building.
    -   `services/`: Application services that encapsulate specific business operations, such as roof measurement calculations.
    -   `ui/`: User interface components, views, and presentation logic using PySide6. Includes the main window, menu/tool/status bars, and the interactive roof canvas.
    -   `utils/`: General utility functions and helper classes used across the application (currently integrated within `core`).

-   `assets/`: Stores static assets such as images, icons, and other media files.
-   `config/`: Configuration files for the application, environment settings, and external service credentials.
-   `data/`: Placeholder for application-generated data, temporary files, or cached information (e.g., SQLite database file).
-   `docs/`: Project documentation, API references, and user manuals.
-   `projects/`: Stores project-specific files, user-saved projects, and related data.
-   `tests/`: Contains unit, integration, and end-to-end tests for the application, implemented with `pytest`.

## Installation

To set up and run Roof AI Studio locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/RoofAIStudio.git
    cd RoofAIStudio
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

After installation, you can run the application:

1.  **Activate your virtual environment** (if not already active).
2.  **Run the main application script:**
    ```bash
    python main.py
    ```

The application window should appear.

## Development

### Running Tests

To run the test suite:

1.  **Activate your virtual environment.**
2.  **Navigate to the project root directory.**
3.  **Run pytest:**
    ```bash
    pytest
    ```

### Development Roadmap (v0.1.0 - Initial Architecture Release)

This initial release focuses on establishing a robust and extensible architectural foundation. Future development will build upon this core.

**Key areas for future development:**

*   **Enhanced UI/UX:** Implement detailed panels for project info, properties, measurements, and material lists.
*   **Project Management:** Functionality to create, save, load, and manage projects (integrating with the `database` module).
*   **Advanced Geometry Editor:** Tools for editing multiple roof planes, adding ridges, valleys, and openings.
*   **Calibration Tools:** Interactive tools for users to define real-world scale on images.
*   **AI Model Integration:** Replace placeholder AI models with actual trained YOLO/SAM models for object detection and segmentation.
*   **3D Visualization:** Integrate 3D rendering of roof geometry.
*   **Material Database Management:** UI for managing material categories, manufacturers, and individual materials.
*   **Pricing Configuration:** UI for setting up labor rates, price rules, and generating detailed estimates.
*   **Plugin System:** Develop a functional plugin loading and management system.
*   **Undo/Redo Functionality:** Implement a robust undo/redo stack for geometry editing.
*   **Localization:** Support for multiple languages.
