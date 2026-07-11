"""
Tests for the app.materials module.
"""

import pytest
import uuid
from math import isclose

from app.materials.material_model import Material, MaterialCategory, MaterialManufacturer
from app.materials.material_repository import SQLAlchemyMaterialRepository
from app.materials.material_service import MaterialService
from app.materials.calculation_result import MaterialCalculationResult
from app.materials.material_calculator import MaterialCalculator
from app.database.enums import MaterialUnit
from app.database.models.material import Material as ORMMaterial, MaterialCategory as ORMMaterialCategory, MaterialManufacturer as ORMMaterialManufacturer
from app.services.measurement_service import RoofMeasurementResult

class TestMaterialModels:
    def test_material_creation(self):
        category = MaterialCategory(id=uuid.uuid4(), name="Shingles")
        manufacturer = MaterialManufacturer(id=uuid.uuid4(), name="Acme")
        material = Material(
            id=uuid.uuid4(),
            name="Asphalt Shingle",
            category=category,
            manufacturer=manufacturer,
            unit=MaterialUnit.SQUARE_FOOT,
            price=1.5,
            coverage=100.0,
            waste_factor=0.10
        )
        assert material.name == "Asphalt Shingle"
        assert material.category.name == "Shingles"
        assert material.unit == MaterialUnit.SQUARE_FOOT
        assert isclose(material.price, 1.5)
        assert isclose(material.coverage, 100.0)
        assert isclose(material.waste_factor, 0.10)

    def test_material_invalid_price(self):
        category = MaterialCategory(id=uuid.uuid4(), name="Shingles")
        with pytest.raises(ValueError, match="price cannot be negative"):
            Material(
                id=uuid.uuid4(), name="Bad Material", category=category, manufacturer=None,
                unit=MaterialUnit.EACH, price=-1.0, coverage=1.0, waste_factor=0.0
            )

    def test_material_invalid_waste_factor(self):
        category = MaterialCategory(id=uuid.uuid4(), name="Shingles")
        with pytest.raises(ValueError, match="Waste factor must be between 0 and 1"):
            Material(
                id=uuid.uuid4(), name="Bad Material", category=category, manufacturer=None,
                unit=MaterialUnit.EACH, price=1.0, coverage=1.0, waste_factor=1.1
            )

class TestSQLAlchemyMaterialRepository:
    def test_add_get_material(self, material_repository: SQLAlchemyMaterialRepository, sample_material_category: ORMMaterialCategory, sample_material_manufacturer: ORMMaterialManufacturer):
        domain_category = material_repository._orm_to_domain_category(sample_material_category)
        domain_manufacturer = material_repository._orm_to_domain_manufacturer(sample_material_manufacturer)
        
        new_material = Material(
            id=uuid.uuid4(),
            name="Test Shingle",
            category=domain_category,
            manufacturer=domain_manufacturer,
            unit=MaterialUnit.SQUARE_FOOT,
            price=2.0,
            coverage=100.0,
            waste_factor=0.05
        )
        added_material = material_repository.add_material(new_material)
        assert added_material.id == new_material.id
        assert added_material.name == "Test Shingle"

        retrieved_material = material_repository.get_material(added_material.id)
        assert retrieved_material is not None
        assert retrieved_material.name == "Test Shingle"
        assert retrieved_material.category.name == "Shingles"

    def test_update_material(self, material_repository: SQLAlchemyMaterialRepository, sample_material: ORMMaterial):
        retrieved_material = material_repository.get_material(sample_material.id)
        assert retrieved_material is not None
        
        updated_material = Material(
            id=retrieved_material.id,
            name="Updated Shingle",
            category=retrieved_material.category,
            manufacturer=retrieved_material.manufacturer,
            unit=retrieved_material.unit,
            price=3.0,
            coverage=retrieved_material.coverage,
            waste_factor=retrieved_material.waste_factor,
            sku="UPD-001",
            is_active=False
        )
        material_repository.update_material(updated_material)
        
        checked_material = material_repository.get_material(updated_material.id)
        assert checked_material.name == "Updated Shingle"
        assert isclose(checked_material.price, 3.0)
        assert checked_material.sku == "UPD-001"
        assert checked_material.is_active is False

    def test_delete_material(self, material_repository: SQLAlchemyMaterialRepository, sample_material: ORMMaterial):
        material_repository.delete_material(sample_material.id)
        deleted_material = material_repository.get_material(sample_material.id)
        assert deleted_material is None

    def test_search_materials(self, material_repository: SQLAlchemyMaterialRepository, sample_material: ORMMaterial):
        found_materials = material_repository.search_materials(name="Asphalt")
        assert len(found_materials) == 1
        assert found_materials[0].name == "Asphalt Shingle"

        found_materials_by_category = material_repository.search_materials(category_id=sample_material.category_id)
        assert len(found_materials_by_category) == 1

        found_materials_inactive = material_repository.search_materials(is_active=False)
        assert len(found_materials_inactive) == 0

class TestMaterialService:
    @pytest.fixture
    def mock_material_repository(self, mocker):
        repo = mocker.Mock(spec=SQLAlchemyMaterialRepository)
        # Mock a material for calculations
        repo.get_material.return_value = Material(
            id=uuid.uuid4(),
            name="Test Shingle",
            category=MaterialCategory(id=uuid.uuid4(), name="Shingles"),
            manufacturer=None,
            unit=MaterialUnit.SQUARE_FOOT,
            price=1.0,
            coverage=10.0, # 10 sq ft per unit
            waste_factor=0.10
        )
        return repo

    def test_calculate_material_quantity_area(self, mock_material_repository):
        service = MaterialService(mock_material_repository)
        material_id = uuid.uuid4()
        # 100 sq m = 1076.39 sq ft
        # 1076.39 sq ft / 10 sq ft/unit = 107.639 units
        quantity = service.calculate_material_quantity(material_id, required_area_sq_m=100.0)
        assert isclose(quantity, 107.63910416709722)

    def test_apply_waste_factor(self, mock_material_repository):
        service = MaterialService(mock_material_repository)
        material_id = uuid.uuid4()
        base_quantity = 100.0
        # 100 * (1 + 0.10) = 110
        total_quantity = service.apply_waste_factor(base_quantity, material_id)
        assert isclose(total_quantity, 110.0)

    def test_calculate_cost(self, mock_material_repository):
        service = MaterialService(mock_material_repository)
        material_id = uuid.uuid4()
        quantity = 110.0
        # 110 * 1.0 = 110.0
        cost = service.calculate_cost(material_id, quantity)
        assert isclose(cost, 110.0)

class TestMaterialCalculator:
    @pytest.fixture
    def mock_material_repository_for_calculator(self, mocker):
        repo = mocker.Mock(spec=SQLAlchemyMaterialRepository)
        # Define a few materials for the calculator
        repo.get_material.side_effect = lambda mat_id: {
            uuid.UUID("00000000-0000-0000-0000-000000000001"): Material(
                id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                name="Covering Tile",
                category=MaterialCategory(id=uuid.uuid4(), name="Covering"),
                manufacturer=None,
                unit=MaterialUnit.SQUARE_FOOT,
                price=2.0,
                coverage=5.0, # 5 sq ft per unit
                waste_factor=0.10
            ),
            uuid.UUID("00000000-0000-0000-0000-000000000002"): Material(
                id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                name="Underlayment Membrane",
                category=MaterialCategory(id=uuid.uuid4(), name="Underlayment"),
                manufacturer=None,
                unit=MaterialUnit.SQUARE_FOOT,
                price=0.5,
                coverage=20.0, # 20 sq ft per unit
                waste_factor=0.05
            ),
            uuid.UUID("00000000-0000-0000-0000-000000000003"): Material(
                id=uuid.UUID("00000000-0000-0000-0000-000000000003"),
                name="Batten Wood",
                category=MaterialCategory(id=uuid.uuid4(), name="Wood"),
                manufacturer=None,
                unit=MaterialUnit.LINEAR_FOOT,
                price=1.0,
                coverage=1.0, # 1 linear ft per unit
                waste_factor=0.08
            )
        }.get(mat_id)
        return repo

    @pytest.fixture
    def sample_roof_measurement_result(self) -> RoofMeasurementResult:
        return RoofMeasurementResult(
            total_area_m2=100.0, # 100 sq m
            total_perimeter_m=40.0, # 40 m
            plane_areas_m2={"Main Plane": 100.0},
            edge_lengths_m={}
        )

    def test_calculate_covering(self, mock_material_repository_for_calculator, sample_roof_measurement_result):
        calculator = MaterialCalculator(mock_material_repository_for_calculator)
        covering_material_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
        
        result = calculator.calculate_covering(sample_roof_measurement_result, covering_material_id)
        
        assert isinstance(result, MaterialCalculationResult)
        assert result.material_id == covering_material_id
        # 100 sq m = 1076.39 sq ft
        # Base quantity = 1076.39 / 5 = 215.278
        # Waste quantity = 215.278 * 0.10 = 21.5278
        # Total quantity = 215.278 + 21.5278 = 236.8058
        # Estimated cost = 236.8058 * 2.0 = 473.6116
        assert isclose(result.quantity, 215.27820833419445)
        assert isclose(result.waste_quantity, 21.527820833419445)
        assert isclose(result.total_quantity, 236.8060291676139)
        assert isclose(result.estimated_cost, 473.6120583352278)

    def test_calculate_membrane(self, mock_material_repository_for_calculator, sample_roof_measurement_result):
        calculator = MaterialCalculator(mock_material_repository_for_calculator)
        membrane_material_id = uuid.UUID("00000000-0000-0000-0000-000000000002")
        
        result = calculator.calculate_membrane(sample_roof_measurement_result, membrane_material_id)
        
        assert isinstance(result, MaterialCalculationResult)
        assert result.material_id == membrane_material_id
        # 100 sq m = 1076.39 sq ft
        # Base quantity = 1076.39 / 20 = 53.8195
        # Waste quantity = 53.8195 * 0.05 = 2.690975
        # Total quantity = 53.8195 + 2.690975 = 56.510475
        # Estimated cost = 56.510475 * 0.5 = 28.2552375
        assert isclose(result.quantity, 53.81955208354861)
        assert isclose(result.waste_quantity, 2.6909776041774306)
        assert isclose(result.total_quantity, 56.51052968772604)
        assert isclose(result.estimated_cost, 28.25526484386302)

    def test_calculate_battens(self, mock_material_repository_for_calculator, sample_roof_measurement_result):
        calculator = MaterialCalculator(mock_material_repository_for_calculator)
        batten_material_id = uuid.UUID("00000000-0000-0000-0000-000000000003")
        
        result = calculator.calculate_battens(sample_roof_measurement_result, batten_material_id)
        
        assert isinstance(result, MaterialCalculationResult)
        assert result.material_id == batten_material_id
        # Required linear m = 40.0 * 1.5 = 60.0 m
        # 60.0 m = 196.85 linear ft
        # Base quantity = 196.85 / 1.0 = 196.85
        # Waste quantity = 196.85 * 0.08 = 15.748
        # Total quantity = 196.85 + 15.748 = 212.598
        # Estimated cost = 212.598 * 1.0 = 212.598
        assert isclose(result.quantity, 196.8503937007874)
        assert isclose(result.waste_quantity, 15.748031496062992)
        assert isclose(result.total_quantity, 212.5984251968504)
        assert isclose(result.estimated_cost, 212.5984251968504)

    def test_calculate_total_cost(self):
        results = [
            MaterialCalculationResult(uuid.uuid4(), "Mat1", 10, MaterialUnit.EACH, 1, 11, 110.0),
            MaterialCalculationResult(uuid.uuid4(), "Mat2", 5, MaterialUnit.EACH, 0.5, 5.5, 55.0)
        ]
        total_cost = MaterialCalculator(None).calculate_total_cost(results) # Repo not needed for this method
        assert isclose(total_cost, 165.0)
