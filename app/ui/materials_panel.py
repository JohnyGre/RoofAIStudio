"""Panel pre vyber streneho materialu, dodavatela a vypocet ceny."""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QFrame, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Signal, Qt
import uuid

from app.database.session import get_db_session
from app.pricing.roof_pricing_service import RoofPricingService, MaterialPriceResult


class MaterialsPanel(QFrame):
    """Panel for material selection, supplier, and price calculation."""
    material_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MaterialsPanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setMinimumWidth(220)
        self.setMaximumHeight(350)
        self.setStyleSheet(
            "#MaterialsPanel { background-color: #3E5060; border: 1px solid #2C3E50; border-radius: 5px; }"
            " QLabel { color: #ECF0F1; } QComboBox { background: #4a6074; color: #ECF0F1; border: 1px solid #2C3E50; padding: 3px; }"
            " QPushButton { background: #2980b9; color: white; border: none; padding: 6px 12px; border-radius: 3px; }"
            " QPushButton:hover { background: #3498db; }"
        )
        self._pricing = None
        self._roof_area_m2 = 0.0
        self._last_result = None
        self._slope_deg = 25.0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        title = QLabel("Materialy a cena")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Category filter
        cat_layout = QHBoxLayout()
        cat_layout.addWidget(QLabel("Kategoria:"))
        self._cat_combo = QComboBox()
        self._cat_combo.addItem("-- Vsetky --", None)
        self._cat_combo.currentIndexChanged.connect(self._on_filter_changed)
        cat_layout.addWidget(self._cat_combo)
        layout.addLayout(cat_layout)

        # Material selection
        mat_layout = QHBoxLayout()
        mat_layout.addWidget(QLabel("Material:"))
        self._mat_combo = QComboBox()
        self._mat_combo.addItem("-- Vyber --", None)
        self._mat_combo.currentIndexChanged.connect(self._on_material_changed)
        mat_layout.addWidget(self._mat_combo)
        layout.addLayout(mat_layout)

        # Supplier selection
        sup_layout = QHBoxLayout()
        sup_layout.addWidget(QLabel("Dodavatel:"))
        self._sup_combo = QComboBox()
        self._sup_combo.addItem("-- Vyber --", None)
        self._sup_combo.currentIndexChanged.connect(self._on_supplier_changed)
        sup_layout.addWidget(self._sup_combo)
        layout.addLayout(sup_layout)

        # Price labels
        self._price_label = QLabel("Cena: -- EUR/m2")
        self._price_label.setStyleSheet("font-size: 12px; color: #f1c40f;")
        layout.addWidget(self._price_label)
        self._total_label = QLabel("Celkova cena: -- EUR")
        self._total_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #2ecc71;")
        layout.addWidget(self._total_label)
        self._info_label = QLabel("Plocha: -- m2 | Sklon: --")
        self._info_label.setStyleSheet("font-size: 10px; color: #aab;")
        layout.addWidget(self._info_label)
        self._waste_label = QLabel("Odpad: -- m2 | Celkom: -- m2")
        self._waste_label.setStyleSheet("font-size: 10px; color: #aab;")
        layout.addWidget(self._waste_label)
        layout.addStretch(1)

    def init_pricing(self):
        session = next(get_db_session())
        self._pricing = RoofPricingService(session)
        self._refresh_materials()

    def set_roof_data(self, area_m2: float, slope_deg: float):
        self._roof_area_m2 = area_m2
        self._slope_deg = slope_deg
        self._update_display()

    def _refresh_materials(self):
        if not self._pricing:
            return
        self._mat_combo.blockSignals(True)
        self._mat_combo.clear()
        self._mat_combo.addItem("-- Vyber --", None)
        cat_val = self._cat_combo.currentData()
        materials = self._pricing.get_materials_by_slope(self._slope_deg)
        for m in materials:
            if cat_val and m.category.name != cat_val:
                continue
            cat_name = m.category.name if m.category else "?"
            display = m.name + " (" + cat_name + ")"
            self._mat_combo.addItem(display, m.id)
        self._mat_combo.blockSignals(False)

    def _refresh_suppliers(self):
        if not self._pricing:
            return
        mat_id = self._mat_combo.currentData()
        self._sup_combo.blockSignals(True)
        self._sup_combo.clear()
        self._sup_combo.addItem("-- Vyber --", None)
        if mat_id:
            for s in self._pricing.get_suppliers_for_material(mat_id):
                display = s["supplier_name"] + " (" + str(s["price_per_unit"]) + " EUR/m2)"
                self._sup_combo.addItem(display, s["supplier_id"])
        self._sup_combo.blockSignals(False)

    def _on_filter_changed(self, _idx):
        self._refresh_materials()
        self._update_display()

    def _on_material_changed(self, _idx):
        self._refresh_suppliers()
        self._update_display()

    def _on_supplier_changed(self, _idx):
        self._update_display()

    def _update_display(self):
        mat_id = self._mat_combo.currentData()
        sup_id = self._sup_combo.currentData()
        self._info_label.setText("Plocha: " + str(round(self._roof_area_m2, 1)) + " m2 | Sklon: " + str(round(self._slope_deg)) + chr(176))
        if not mat_id or not self._pricing or self._roof_area_m2 <= 0:
            self._price_label.setText("Cena: -- EUR/m2")
            self._total_label.setText("Celkova cena: -- EUR")
            self._waste_label.setText("Odpad: -- m2 | Celkom: -- m2")
            return
        result = self._pricing.calculate_price(mat_id, sup_id, self._roof_area_m2)
        if result:
            self._price_label.setText("Cena: " + str(round(result.price_per_m2, 2)) + " EUR/m2 (" + result.supplier_name + ")")
            self._total_label.setText("Celkova cena: " + str(round(result.total_price_eur, 2)) + " EUR")
            self._waste_label.setText(
                "Odpad (" + str(int(result.waste_factor*100)) + "%): " + str(round(result.waste_area_m2, 1)) + " m2 | Celkom: " + str(round(result.total_area_m2, 1)) + " m2"
            )
            self._last_result = result
            self.material_changed.emit(result)
        else:
            self._price_label.setText("Cena: -- EUR/m2")
            self._total_label.setText("Celkova cena: -- EUR")