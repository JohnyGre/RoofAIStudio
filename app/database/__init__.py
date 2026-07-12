from .base import Base, BaseModel
from .enums import (
    RoofType, RoofStatus, EstimateStatus, MaterialUnit, PhotoType, AIModelType, EdgeType
)
from .models.customer import Customer
from .models.project import Project
from .models.roof import Roof, RoofPhoto
from .models.geometry import RoofGeometry, RoofPlane, RoofEdge, RoofVertex
from .models.material import Material, MaterialCategory, MaterialManufacturer
from .models.estimate import Estimate, EstimateItem
from .models.roof_template import RoofTemplate, RoofTemplatePlane
from .models.ai import AIModel, AIPrediction
from .models.training_sample import AITrainingSample
from .models.settings import ApplicationSettings
from .models.supplier import Supplier
from .models.price_list import PriceList, PriceItem
from .models.price_history import PriceHistory
from .models.labor import LaborPrice
from .models.roof_knowledge import RoofType, RoofTopology, RoofGeometryRules, RoofFeature # Updated import
from .models.roof_genome_data import RoofGenome # New import for the actual genome data
from .database import db_manager
from .session import get_db_session