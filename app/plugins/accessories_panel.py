#!/usr/bin/env python3
"""Accessories Panel — view/search roof accessories from cennik_doplnky.csv."""
import os, sys
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QLabel, QComboBox, QPushButton
)
from PySide6.QtCore import Qt

# Resolve project root
_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(_PROJECT, "data", "cennik_doplnky.csv")


class AccessoriesPanel(QDialog):
    """Searchable accessories table with filtering."""
    
    COLUMNS = [
        ("Kategória", "category"),
        ("Výrobca", "manufacturer"),
        ("Názov", "name"),
        ("Jednotka", "unit"),
        ("Cena", "price"),
        ("Mena", "currency"),
        ("Dodávateľ", "supplier"),
    ]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Strešné doplnky — Cenník")
        self.setMinimumSize(1000, 550)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._data = []
        
        layout = QVBoxLayout(self)
        
        # Filter bar
        filter_layout = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Hľadať podľa názvu / výrobcu / dodávateľa...")
        self._search.textChanged.connect(self._apply_filters)
        filter_layout.addWidget(self._search)
        
        self._cat_filter = QComboBox()
        self._cat_filter.addItem("Všetky kategórie")
        self._cat_filter.currentTextChanged.connect(self._apply_filters)
        filter_layout.addWidget(self._cat_filter)
        
        self._mfr_filter = QComboBox()
        self._mfr_filter.addItem("Všetci výrobcovia")
        self._mfr_filter.currentTextChanged.connect(self._apply_filters)
        filter_layout.addWidget(self._mfr_filter)
        
        layout.addLayout(filter_layout)
        
        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels([c[0] for c in self.COLUMNS])
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)  # Stretch name
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        layout.addWidget(self._table)
        
        # Load button
        btn_layout = QHBoxLayout()
        self._reload_btn = QPushButton("Znovu načítať CSV")
        self._reload_btn.clicked.connect(self._load_csv)
        btn_layout.addWidget(self._reload_btn)
        btn_layout.addStretch()
        self._count_label = QLabel()
        btn_layout.addWidget(self._count_label)
        layout.addLayout(btn_layout)
        
        self._load_csv()
    
    def _load_csv(self):
        """Parse CSV and populate table."""
        import csv
        self._data = []
        categories = set()
        manufacturers = set()
        
        if not os.path.exists(CSV_PATH):
            self._count_label.setText("CSV nenájdený!")
            return
        
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader((l for l in f if not l.startswith("#")), delimiter=";")
            for row in reader:
                if len(row) < 17:
                    continue
                k, mfr = row[0].strip(), row[1].strip()
                name, unit = row[3].strip(), row[4].strip()
                price_raw = row[13].strip().replace(",", ".")
                currency = row[14].strip() or "EUR"
                supplier = row[10].strip()
                try:
                    price = float(price_raw)
                except ValueError:
                    continue
                if not k or not name:
                    continue
                
                self._data.append({
                    "category": k, "manufacturer": mfr, "name": name,
                    "unit": unit, "price": price, "currency": currency,
                    "supplier": supplier
                })
                categories.add(k)
                manufacturers.add(mfr)
        
        # Update filter combos
        self._cat_filter.blockSignals(True)
        self._cat_filter.clear()
        self._cat_filter.addItem("Všetky kategórie")
        for c in sorted(categories):
            self._cat_filter.addItem(c)
        self._cat_filter.blockSignals(False)
        
        self._mfr_filter.blockSignals(True)
        self._mfr_filter.clear()
        self._mfr_filter.addItem("Všetci výrobcovia")
        for m in sorted(manufacturers):
            self._mfr_filter.addItem(m)
        self._mfr_filter.blockSignals(False)
        
        self._apply_filters()
    
    def _apply_filters(self):
        """Filter data by search text and dropdowns."""
        search = self._search.text().lower()
        cat = self._cat_filter.currentText()
        mfr = self._mfr_filter.currentText()
        
        filtered = self._data
        if cat != "Všetky kategórie":
            filtered = [d for d in filtered if d["category"] == cat]
        if mfr != "Všetci výrobcovia":
            filtered = [d for d in filtered if d["manufacturer"] == mfr]
        if search:
            filtered = [d for d in filtered if (
                search in d["name"].lower() or 
                search in d["manufacturer"].lower() or
                search in d["supplier"].lower() or
                search in d["category"].lower()
            )]
        
        self._table.setRowCount(len(filtered))
        for i, d in enumerate(filtered):
            self._table.setItem(i, 0, QTableWidgetItem(d["category"]))
            self._table.setItem(i, 1, QTableWidgetItem(d["manufacturer"]))
            self._table.setItem(i, 2, QTableWidgetItem(d["name"]))
            self._table.setItem(i, 3, QTableWidgetItem(d["unit"]))
            price_item = QTableWidgetItem(f'{d["price"]:.2f}')
            price_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(i, 4, price_item)
            self._table.setItem(i, 5, QTableWidgetItem(d["currency"]))
            self._table.setItem(i, 6, QTableWidgetItem(d["supplier"]))
        
        self._count_label.setText(f"{len(filtered)} položiek")


def register_plugin(main_window):
    """Register in Tools menu."""
    mb = main_window.menu_bar
    a = mb.tools_menu.addAction("Strešné doplnky / Príslušenstvo...")
    a.triggered.connect(lambda: AccessoriesPanel(main_window).show())
    print("Accessories plugin registered: Tools > Strešné doplnky / Príslušenstvo...")
