"""
Configuration file for pytest, providing common fixtures for tests.
"""

import pytest
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.database.base import Base
from app.database.models.material import Material as ORMMaterial, MaterialCategory as ORMMaterialCategory, MaterialManufacturer as ORMMaterialManufacturer
from app.database.enums import MaterialUnit
from app.materials.material_repository import SQLAlchemyMaterialRepository
from app.geometry.point import Point2D, Point3D
from app.geometry.edge import Edge
from app.geometry.polygon import Polygon2D
from app.geometry.plane import RoofPlane
from app.geometry.roof_geometry import RoofGeometry
from app.geometry.calibration import CalibrationModel
from app.ai.ai_model import AIModel
from app.ai.ai_result import DetectionResult, BoundingBox
from app.ai.models.roof_detector import RoofDetector
from app.ai.model_registry import model_registry
from app.ai.ai_engine import AIEngine
from app.ai.geometry_converter import GeometryConverter
from app.ai.analysis_pipeline import RoofAnalysisPipeline
from app.core.image.image_loader import ImageLoader
from app.core.image.image_processor import ImageProcessor
from app.core.config import config # Import the config object
from app.core.app_paths import app_paths # Import app_paths for test paths

# Ensure test directories exist
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    test_data_dir = app_paths.base_dir / "tests" / "data"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    # Create a dummy image for testing
    dummy_image_path = test_data_dir / "dummy_roof.jpg"
    if not dummy_image_path.exists():
        try:
            from PIL import Image as PILImage
            img = PILImage.new('RGB', (800, 600), color = 'red')
            img.save(dummy_image_path)
        except ImportError:
            print("Pillow not installed, cannot create dummy image. Some tests might fail.")
    yield
    # Teardown: clean up dummy image if created
    if dummy_image_path.exists():
        dummy_image_path.unlink()

@pytest.fixture(scope="function")
def db_session():
    """
    Provides a SQLAlchemy session for testing.
    Uses an in-memory SQLite database for isolation.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine) # Clean up tables after each test

@pytest.fixture(scope="function")
def material_repository(db_session: Session) -> SQLAlchemyMaterialRepository:
    """
    Provides an instance of SQLAlchemyMaterialRepository with a test session.
    """
    return SQLAlchemyMaterialRepository(db_session)

@pytest.fixture
def sample_material_category(db_session: Session) -> ORMMaterialCategory:
    category = ORMMaterialCategory(id=uuid.uuid4(), name="Shingles", description="Roofing shingles")
    db_session.add(category)
    db_session.commit()
    db_session.refresh(category)
    return category

@pytest.fixture
def sample_material_manufacturer(db_session: Session) -> ORMMaterialManufacturer:
    manufacturer = ORMMaterialManufacturer(id=uuid.uuid4(), name="Acme Inc.", contact_info="info@acme.com")
    db_session.add(manufacturer)
    db_session.commit()
    db_session.refresh(manufacturer)
    return manufacturer

@pytest.fixture
def sample_material(db_session: Session, sample_material_category: ORMMaterialCategory, sample_material_manufacturer: ORMMaterialManufacturer) -> ORMMaterial:
    material = ORMMaterial(
        id=uuid.uuid4(),
        name="Asphalt Shingle",
        category_id=sample_material_category.id,
        manufacturer_id=sample_material_manufacturer.id,
        unit_cost=1.5,
        unit_of_measure=MaterialUnit.SQUARE_FOOT.value,
        sku="ASH-001",
        is_active=True
    )
    db_session.add(material)
    db_session.commit()
    db_session.refresh(material)
    return material

@pytest.fixture
def sample_point2d_1() -> Point2D:
    return Point2D(x=0.0, y=0.0)

@pytest.fixture
def sample_point2d_2() -> Point2D:
    return Point2D(x=3.0, y=4.0)

@pytest.fixture
def sample_point3d_1() -> Point3D:
    return Point3D(x=0.0, y=0.0, z=0.0)

@pytest.fixture
def sample_point3d_2() -> Point3D:
    return Point3D(x=3.0, y=4.0, z=5.0)

@pytest.fixture
def sample_edge() -> Edge:
    return Edge(start_point=Point3D(0,0,0), end_point=Point3D(10,0,0))

@pytest.fixture
def sample_polygon2d() -> Polygon2D:
    return Polygon2D(vertices=[Point2D(0,0), Point2D(10,0), Point2D(10,10), Point2D(0,10)])

@pytest.fixture
def sample_roof_plane() -> RoofPlane:
    return RoofPlane(
        name="Main Plane",
        polygon=Polygon2D(vertices=[Point2D(0,0), Point2D(10,0), Point2D(10,10), Point2D(0,10)]),
        slope=30.0,
        orientation=0.0,
        height_at_vertices=[0.0, 0.0, 5.0, 5.0] # Example for a sloped plane
    )

@pytest.fixture
def sample_roof_geometry(sample_roof_plane: RoofPlane) -> RoofGeometry:
    return RoofGeometry(
        vertices=[Point3D(0,0,0), Point3D(10,0,0), Point3D(10,10,5), Point3D(0,10,5)],
        edges=[Edge(Point3D(0,0,0), Point3D(10,0,0))],
        planes=[sample_roof_plane],
        ridges=[],
        valleys=[],
        openings=[]
    )

@pytest.fixture
def sample_calibration_model() -> CalibrationModel:
    return CalibrationModel(
        reference_points_pixel=(Point2D(0,0), Point2D(100,0)),
        reference_distance_meters=1.0,
        scale_factor_pixels_per_meter=100.0,
        unit="m"
    )

@pytest.fixture
def sample_image_path() -> Path:
    # Use the dummy image created by setup_test_environment
    return app_paths.base_dir / "tests" / "data" / "dummy_roof.jpg"

@pytest.fixture
def sample_image_data(sample_image_path: Path) -> np.ndarray:
    # Ensure the dummy image exists before trying to load it
    if not sample_image_path.exists():
        from PIL import Image as PILImage
        img = PILImage.new('RGB', (800, 600), color = 'red')
        img.save(sample_image_path)
    
    img_data = ImageLoader.load_image(sample_image_path, as_opencv=True)
    if img_data is None:
        pytest.fail(f"Failed to load sample image data from {sample_image_path}")
    return img_data

@pytest.fixture
def ai_engine() -> AIEngine:
    # Clear registry to ensure clean state for each test
    model_registry.clear_registry()
    engine = AIEngine()
    # Register a default detector for pipeline tests
    engine.register_model(RoofDetector())
    return engine

@pytest.fixture
def geometry_converter() -> GeometryConverter:
    return GeometryConverter()

@pytest.fixture
def roof_analysis_pipeline(ai_engine: AIEngine, geometry_converter: GeometryConverter) -> RoofAnalysisPipeline:
    return RoofAnalysisPipeline(ai_engine, geometry_converter)

@pytest.fixture
def sample_detection_result() -> DetectionResult:
    return DetectionResult(
        bounding_box=BoundingBox(x_min=10.0, y_min=20.0, x_max=110.0, y_max=120.0),
        confidence=0.95,
        class_name="roof_area"
    )
