"""Panel pre vyber streneho materialu, dodavatela a vypocet ceny."""
from datetime import date, datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QFrame, QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem, QHeaderView
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
        self.setMinimumWidth(260)
        self.setMaximumHeight(550)
        self.setStyleSheet(
            "#MaterialsPanel { background-color: #2C3E50; border: 1px solid #1a252f; border-radius: 5px; }"
            " QLabel { color: #ECF0F1; } QComboBox { background: #34495e; color: #ECF0F1; border: 1px solid #1a252f; padding: 3px; }"
            " QPushButton { background: #2980b9; color: white; border: none; padding: 6px 12px; border-radius: 3px; }"
            " QPushButton:hover { background: #3498db; }"
            " QTableWidget { background: #34495e; color: #ECF0F1; gridline-color: #1a252f; border: 1px solid #1a252f; }"
            " QTableWidget::item { padding: 2px 4px; }"
            " QHeaderView::section { background: #1a252f; color: #ECF0F1; padding: 3px; border: 1px solid #2C3E50; }"
        )
        self._pricing = None
        self._roof_area_m2 = 0.0
        self._last_result = None
        self._slope_deg = 25.0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)

        # Title
        title = QLabel("Materialy a cena")
        title.setStyleSheet("font-weight: bold; font-size: 13px; color: #3498db;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Category
        cat_layout = QHBoxLayout()
        cat_layout.addWidget(QLabel("Kategoria:"))
        self._cat_combo = QComboBox()
        self._cat_combo.addItem("-- Vsetky --", None)
        self._cat_combo.currentIndexChanged.connect(self._on_filter_changed)
        cat_layout.addWidget(self._cat_combo)
        layout.addLayout(cat_layout)

        # Material
        mat_layout = QHBoxLayout()
        mat_layout.addWidget(QLabel("Material:"))
        self._mat_combo = QComboBox()
        self._mat_combo.addItem("-- Vyber --", None)
        self._mat_combo.currentIndexChanged.connect(self._on_material_changed)
        mat_layout.addWidget(self._mat_combo)
        layout.addLayout(mat_layout)

        # Supplier
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
        self._info_label.setStyleSheet("font-size: 10px; color: #95a5a6;")
        layout.addWidget(self._info_label)

        self._waste_label = QLabel("Odpad: -- m2 | Celkom: -- m2")
        self._waste_label.setStyleSheet("font-size: 10px; color: #95a5a6;")
        layout.addWidget(self._waste_label)

        # Date label (platnost ceny)
        self._date_label = QLabel("")
        self._date_label.setStyleSheet("font-size: 11px;")
        self._date_label.setWordWrap(True)
        layout.addWidget(self._date_label)

        # Separator and comparison title
        sep = QLabel("")
        sep.setFixedHeight(6)
        layout.addWidget(sep)

        comp_title = QLabel("Porovnanie dodavatelov:")
        comp_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #3498db; margin-top: 4px;")
        layout.addWidget(comp_title)

        # Comparison table
        self._comp_table = QTableWidget(0, 4)
        self._comp_table.setHorizontalHeaderLabels(["Dodavatel", "Cena/m2", "Platna od", "Celkova cena"])
        self._comp_table.horizontalHeader().setStretchLastSection(True)
        self._comp_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._comp_table.setMaximumHeight(160)
        self._comp_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._comp_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._comp_table.verticalHeader().setVisible(False)
        layout.addWidget(self._comp_table)

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
                platnost = s.get("platnost_od", "")
                date_str = f" (od {platnost})" if platnost else ""
                display = s["supplier_name"] + " (" + str(s["price_per_unit"]) + " EUR/m2)" + date_str
                self._sup_combo.addItem(display, s["supplier_id"])
        self._sup_combo.blockSignals(False)

    def _on_filter_changed(self, _idx):
        self._refresh_materials()
        self._update_display()

    def _on_material_changed(self, _idx):
        self._refresh_suppliers()
        self._update_display()
        self._refresh_comparison()

    def _on_supplier_changed(self, _idx):
        self._update_display()

    def _update_display(self):
        mat_id = self._mat_combo.currentData()
        sup_id = self._sup_combo.currentData()
        self._info_label.setText(
            "Plocha: " + str(round(self._roof_area_m2, 1)) + " m2 | Sklon: " + str(round(self._slope_deg)) + chr(176))
        if not mat_id or not self._pricing or self._roof_area_m2 <= 0:
            self._price_label.setText("Cena: -- EUR/m2")
            self._total_label.setText("Celkova cena: -- EUR")
            self._waste_label.setText("Odpad: -- m2 | Celkom: -- m2")
            self._date_label.setText("")
            return
        result = self._pricing.calculate_price(mat_id, sup_id, self._roof_area_m2)
        if result:
            self._price_label.setText(
                "Cena: " + str(round(result.price_per_m2, 2)) + " EUR/m2 (" + result.supplier_name + ")")
            self._total_label.setText("Celkova cena: " + str(round(result.total_price_eur, 2)) + " EUR")
            self._waste_label.setText(
                "Odpad (" + str(int(result.waste_factor * 100)) + "%): " + str(round(result.waste_area_m2, 1))
                + " m2 | Celkom: " + str(round(result.total_area_m2, 1)) + " m2"
            )

            # Date display with aging
            platnost = result.platnost_od
            if platnost:
                try:
                    d = datetime.strptime(platnost, '%Y-%m-%d').date()
                    age_months = (date.today().year - d.year) * 12 + (date.today().month - d.month)
                    formatted = d.strftime('%d.%m.%Y')
                    if age_months > 6:
                        self._date_label.setText(f"Cena platna od {formatted} (STARA CENA - overit u dodavatela!)")
                        self._date_label.setStyleSheet("font-size: 11px; color: #e67e22; font-weight: bold;")
                    elif age_months > 3:
                        self._date_label.setText(f"Cena platna od {formatted}")
                        self._date_label.setStyleSheet("font-size: 11px; color: #f1c40f;")
                    else:
                        self._date_label.setText(f"Cena platna od {formatted}")
                        self._date_label.setStyleSheet("font-size: 11px; color: #2ecc71;")
                except (ValueError, TypeError):
                    self._date_label.setText("")
                    self._date_label.setStyleSheet("")
            else:
                self._date_label.setText("")
                self._date_label.setStyleSheet("")
            self._last_result = result
            self.material_changed.emit(result)
        else:
            self._price_label.setText("Cena: -- EUR/m2")
            self._total_label.setText("Celkova cena: -- EUR")

    def _refresh_comparison(self):
        """Populate supplier comparison table for selected material."""
        mat_id = self._mat_combo.currentData()
        self._comp_table.setRowCount(0)

        if not mat_id or not self._pricing:
            return

        suppliers = self._pricing.get_suppliers_for_material(mat_id)
        # Re-sort by price
        suppliers = sorted(suppliers, key=lambda s: s["price_per_unit"])
        self._comp_table.setRowCount(len(suppliers))
        for row, s in enumerate(suppliers):
            name_item = QTableWidgetItem(s["supplier_name"])
            price_item = QTableWidgetItem(str(s["price_per_unit"]).replace(".", ",") + " EUR/m2")
            platnost = s.get("platnost_od", "")
            if platnost:
                try:
                    d = datetime.strptime(platnost, '%Y-%m-%d').date()
                    date_display = d.strftime('%d.%m.%Y')
                except (ValueError, TypeError):
                    date_display = platnost
            else:
                date_display = "--"
            date_item = QTableWidgetItem(date_display)

            # Total for this material+supplier with current roof area
            area = self._roof_area_m2 if self._roof_area_m2 > 0 else 100.0
            waste_factor = self._last_result.waste_factor if self._last_result else 0.10
            total = s["price_per_unit"] * area * (1 + waste_factor)
            total_item = QTableWidgetItem(str(round(total, 2)).replace(".", ",") + " EUR")

            self._comp_table.setItem(row, 0, name_item)
            self._comp_table.setItem(row, 1, price_item)
            self._comp_table.setItem(row, 2, date_item)
            self._comp_table.setItem(row, 3, total_item)

            # Highlight cheapest supplier
            if row == 0:
                for col in range(4):
                    it = self._comp_table.item(row, col)
                    if it:
                        it.setBackground(Qt.GlobalColor.darkGreen)
                        it.setForeground(Qt.GlobalColor.white)
