"""
This module provides the PDFExporter service for generating professional PDF reports.
"""

from pathlib import Path
from typing import List, Any, Optional
from datetime import date

from reportlab.platypus import Paragraph, Spacer, PageBreak, Image, Table, TableStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.colors import black, lightgrey, gray, white

from app.exporters.pdf_models import CompanyInfo, CustomerReport
from app.exporters.report_template import ReportTemplate
from app.core.logger import setup_logging

logger = setup_logging()

class PDFExporter:
    """
    Service for generating various types of PDF documents, such as offers and detailed reports.
    It uses ReportLab for document creation and a ReportTemplate for consistent styling.
    """

    def __init__(self):
        pass

    def generate_report(self, report_data: CustomerReport, company_info: CompanyInfo, output_path: Path) -> None:
        """
        Generates a comprehensive customer report in PDF format.

        Args:
            report_data (CustomerReport): The data model containing all information for the report.
            company_info (CompanyInfo): The company's branding information.
            output_path (Path): The file path where the PDF report will be saved.
        """
        logger.info(f"Generating customer report to: {output_path}")
        
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)

        report_template = ReportTemplate(output_path, title=f"{company_info.company_name} - Roof Analysis Report")
        story = report_template.get_story_elements(report_data, company_info)
        
        try:
            report_template.build_document(story)
            logger.info(f"Customer report successfully generated to {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate customer report: {e}")
            raise

    def generate_offer(self, report_data: CustomerReport, company_info: CompanyInfo, output_path: Path) -> None:
        """
        Generates a simplified offer document in PDF format, primarily focusing on the estimate.

        Args:
            report_data (CustomerReport): The data model containing estimate information.
            company_info (CompanyInfo): The company's branding information.
            output_path (Path): The file path where the PDF offer will be saved.
        """
        logger.info(f"Generating offer document to: {output_path}")

        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)

        report_template = ReportTemplate(output_path, title=f"{company_info.company_name} - Offer")
        styles = report_template.styles
        story: List[Any] = []

        # Cover Page for Offer
        story.append(Paragraph(f"{company_info.company_name} - Offer", styles['TitleStyle']))
        story.append(Spacer(1, 0.5 * inch))
        if company_info.logo_path and company_info.logo_path.exists():
            try:
                logo = Image(str(company_info.logo_path), width=2*inch, height=1*inch)
                logo.hAlign = 'CENTER'
                story.append(logo)
                story.append(Spacer(1, 0.5 * inch))
            except Exception as e:
                logger.warning(f"Could not load logo image for offer: {e}")
        story.append(Paragraph(f"Date: {report_data.report_date.strftime('%Y-%m-%d')}", styles['h2']))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(f"To: {report_data.customer_name}", styles['h2']))
        story.append(Paragraph(f"Project: {report_data.project_name}", styles['h2']))
        story.append(PageBreak())

        # Estimate Section
        story.append(Paragraph("Your Roof Estimate", styles['Heading1']))
        if report_data.estimate:
            estimate = report_data.estimate
            story.append(Paragraph(f"<b>Estimate Name:</b> {estimate.name}", styles['BodyText']))
            story.append(Spacer(1, 0.1 * inch))

            # Material Lines
            if estimate.material_lines:
                story.append(Paragraph("<b>Material Costs:</b>", styles['Heading2']))
                mat_data = [['Description', 'Qty', 'Unit Price', 'Total']]
                for line in estimate.material_lines:
                    mat_data.append([line.description, f"{line.quantity:.2f}", f"${line.unit_price:.2f}", f"${line.total_price:.2f}"])
                mat_table = Table(mat_data, colWidths=[5*cm, 2*cm, 2.5*cm, 2.5*cm])
                mat_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, gray),
                ]))
                story.append(mat_table)
                story.append(Spacer(1, 0.1 * inch))

            # Labor Lines
            if estimate.labor_lines:
                story.append(Paragraph("<b>Labor Costs:</b>", styles['Heading2']))
                lab_data = [['Description', 'Qty', 'Unit Price', 'Total']]
                for line in estimate.labor_lines:
                    lab_data.append([line.description, f"{line.quantity:.2f}", f"${line.unit_price:.2f}", f"${line.total_price:.2f}"])
                lab_table = Table(lab_data, colWidths=[5*cm, 2*cm, 2.5*cm, 2.5*cm])
                lab_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, gray),
                ]))
                story.append(lab_table)
                story.append(Spacer(1, 0.1 * inch))

            # Other Lines
            if estimate.other_lines:
                story.append(Paragraph("<b>Other Costs:</b>", styles['Heading2']))
                other_data = [['Description', 'Qty', 'Unit Price', 'Total']]
                for line in estimate.other_lines:
                    other_data.append([line.description, f"{line.quantity:.2f}", f"${line.unit_price:.2f}", f"${line.total_price:.2f}"])
                other_table = Table(other_data, colWidths=[5*cm, 2*cm, 2.5*cm, 2.5*cm])
                other_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, gray),
                ]))
                story.append(other_table)
                story.append(Spacer(1, 0.1 * inch))

            # Summary Table
            summary_data = [
                ['Subtotal:', f"${estimate.subtotal:.2f}"],
                ['Discount:', f"-${estimate.discount:.2f}"],
                ['Price after Discount:', f"${estimate.subtotal - estimate.discount:.2f}"],
                ['Margin:', f"{estimate.margin*100:.0f}%"],
                ['Price with Margin:', f"${(estimate.subtotal - estimate.discount) * (1 + estimate.margin):.2f}"],
                ['VAT:', f"{estimate.vat_rate*100:.0f}%"],
                ['<b>FINAL PRICE:</b>', f"<b>${estimate.final_price:.2f}</b>"]
            ]
            summary_table = Table(summary_data, colWidths=[5*cm, 5*cm])
            summary_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
                ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
                ('BACKGROUND', (0, -1), (-1, -1), lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, gray),
            ]))
            story.append(summary_table)

        else:
            story.append(Paragraph("<i>No price estimate available.</i>", styles['BodyText']))
        story.append(Spacer(1, 0.5 * inch))

        # Signature Area
        story.append(Paragraph(report_data.signature_area_text, styles['Signature']))
        story.append(Spacer(1, 0.5 * inch))
        story.append(Paragraph("_" * 40, styles['Signature'])) # Line for signature
        story.append(Paragraph("Signature", styles['SmallText']))

        try:
            report_template.build_document(story)
            logger.info(f"Offer document successfully generated to {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate offer document: {e}")
            raise
