"""
This module contains styling definitions for the Roof AI Studio application.
"""

class DarkTheme:
    """
    Defines a dark theme for the application.
    """
    # Basic color palette
    PRIMARY_COLOR = "#34495E"  # Dark Blue-Grey
    SECONDARY_COLOR = "#2C3E50" # Even Darker Blue-Grey
    ACCENT_COLOR = "#3498DB"   # Bright Blue
    TEXT_COLOR = "#ECF0F1"     # Light Grey
    HIGHLIGHT_COLOR = "#E74C3C" # Red for warnings/errors

    # Font settings
    FONT_FAMILY = "Segoe UI, Arial, sans-serif"
    FONT_SIZE_NORMAL = "10pt"
    FONT_SIZE_LARGE = "12pt"

    # QSS (Qt Style Sheet) snippets
    MAIN_WINDOW_QSS = f"""
        QMainWindow {{
            background-color: {SECONDARY_COLOR};
            color: {TEXT_COLOR};
        }}
        QMenuBar {{
            background-color: {PRIMARY_COLOR};
            color: {TEXT_COLOR};
            border-bottom: 1px solid {SECONDARY_COLOR};
        }}
        QMenuBar::item {{
            background-color: transparent;
            padding: 5px 10px;
        }}
        QMenuBar::item:selected {{
            background-color: {ACCENT_COLOR};
        }}
        QMenu {{
            background-color: {PRIMARY_COLOR};
            color: {TEXT_COLOR};
            border: 1px solid {SECONDARY_COLOR};
        }}
        QMenu::item {{
            padding: 5px 20px 5px 25px;
        }}
        QMenu::item:selected {{
            background-color: {ACCENT_COLOR};
        }}
        QToolBar {{
            background-color: {PRIMARY_COLOR};
            border-bottom: 1px solid {SECONDARY_COLOR};
            spacing: 5px;
        }}
        QToolButton {{
            color: {TEXT_COLOR};
            background-color: transparent;
            border: none;
            padding: 5px;
        }}
        QToolButton:hover {{
            background-color: {ACCENT_COLOR};
        }}
        QStatusBar {{
            background-color: {PRIMARY_COLOR};
            color: {TEXT_COLOR};
            border-top: 1px solid {SECONDARY_COLOR};
        }}
        QLabel {{
            color: {TEXT_COLOR};
        }}
    """

    # You can add more specific QSS for other widgets here
    # For example:
    # BUTTON_QSS = f"""
    #     QPushButton {{
    #         background-color: {ACCENT_COLOR};
    #         color: {TEXT_COLOR};
    #         border: none;
    #         padding: 8px 15px;
    #         border-radius: 4px;
    #     }}
    #     QPushButton:hover {{
    #         background-color: #5DADE2;
    #     }}
    # """
