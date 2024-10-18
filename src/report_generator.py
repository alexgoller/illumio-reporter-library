from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image, 
    KeepTogether, CondPageBreak, Frame, PageTemplate, Flowable, 
    LongTable, ListFlowable, ListItem, PageBreak  # Add PageBreak here
)
from reportlab.platypus.doctemplate import PageTemplate
import markdown
from xml.etree import ElementTree
import tempfile
import os
import re
import io

class IconAndText(Flowable):
    def __init__(self, icon_path, text, icon_width, icon_height, text_style, icon_position='left'):
        super().__init__()
        self.icon = Image(icon_path, width=icon_width, height=icon_height)
        self.text = text
        self.text_style = text_style
        self.icon_position = icon_position
        self.width = 0
        self.height = 0
        self.paragraph = None

    def wrap(self, availWidth, availHeight):
        # Wrap icon
        icon_width, icon_height = self.icon.wrap(availWidth, availHeight)
        
        # Wrap text
        if self.icon_position == 'left':
            text_width = availWidth - icon_width
        else:  # icon on top
            text_width = availWidth

        self.paragraph = Paragraph(self.text, self.text_style)
        text_width, text_height = self.paragraph.wrap(text_width, availHeight)

        if self.icon_position == 'left':
            self.width = availWidth
            self.height = max(icon_height, text_height)
        else:  # icon on top
            self.width = availWidth
            self.height = icon_height + text_height

        return self.width, self.height

    def draw(self):
        if self.icon_position == 'left':
            self.icon.drawOn(self.canv, 0, self.height - self.icon._height)
            self.paragraph.drawOn(self.canv, self.icon._width, self.height - self.paragraph.height)
        else:  # icon on top
            self.icon.drawOn(self.canv, 0, self.height - self.icon._height)
            self.paragraph.drawOn(self.canv, 0, self.height - self.icon._height - self.paragraph.height)

class ReportGenerator:
    def __init__(self, output_file, color_scheme=None, logo_path=None, header_text=None, footer_text=None):
        self.output_file = output_file
        self.elements = []
        self.styles = getSampleStyleSheet()
        self.current_y = 0
        self.page_height = 792
        self.margin = 72

        # Add these new attributes
        self.color_scheme = color_scheme or {
            'title': colors.black,
            'section': colors.black,
            'text': colors.black,
            'table_header_bg': colors.lightgrey,
            'table_header_text': colors.black,
            'table_body_bg': colors.white,
            'table_body_text': colors.black,
            'table_grid': colors.black
        }
        self.logo_path = logo_path
        self.header_text = header_text
        self.footer_text = footer_text

        # Call setup_styles after initializing color_scheme
        self.setup_styles()

        # Create and add the BulletList style
        bullet_list_style = ParagraphStyle(
            name='BulletList',
            parent=self.styles['Normal'],
            leftIndent=20,
            firstLineIndent=-15,
            spaceBefore=3,
            spaceAfter=3,
            bulletIndent=0,
            bulletFontName='Symbol',
            bulletFontSize=10,
            alignment=TA_LEFT
        )
        self.styles.add(bullet_list_style)

        # Add a bold style
        self.styles.add(ParagraphStyle(name='Bold', parent=self.styles['Normal'], fontName='Helvetica-Bold'))

        self.page_width, self.page_height = letter  # Add this line

    def setup_styles(self):
        self.styles = getSampleStyleSheet()
        try:
            self.styles['Title'].fontSize = 24
            self.styles['Title'].textColor = self.color_scheme['title']
            self.styles['Title'].spaceAfter = 12

            self.styles['Heading2'].fontSize = 18
            self.styles['Heading2'].textColor = self.color_scheme['section']
            self.styles['Heading2'].spaceAfter = 6

            self.styles['Heading3'].fontSize = 14
            self.styles['Heading3'].textColor = self.color_scheme['section']
            self.styles['Heading3'].spaceAfter = 6

            self.styles['Normal'].fontSize = 10
            self.styles['Normal'].textColor = self.color_scheme['text']
            self.styles['Normal'].spaceAfter = 6

        except Exception as e:
            print(f"Error in setup_styles: {e}")
            print(f"Color scheme type: {type(self.color_scheme)}")
            print(f"Color scheme content: {self.color_scheme}")
            # Use default styles if there's an error

    def add_explanation_style(self):
        explanation_style = ParagraphStyle(
            name='Explanation',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=12,
            textColor=self.color_scheme.get('text', colors.black),
            spaceAfter=6
        )
        self.styles.add(explanation_style)

    def add_title(self, title):
        self.elements.append(Paragraph(title, self.styles['Title']))
        self.elements.append(Spacer(1, 12))

    def add_section(self, section_title):
        self.elements.append(Paragraph(section_title, self.styles['Heading2']))
        self.elements.append(Spacer(1, 6))

    def add_table(self, dataframe):
        styles = getSampleStyleSheet()
        normal_style = styles['Normal']

        def to_paragraph(value):
            if isinstance(value, (int, float)):
                return Paragraph(str(value), normal_style)
            elif isinstance(value, str):
                return Paragraph(value, normal_style)
            elif value is None:
                return Paragraph('', normal_style)
            else:
                return Paragraph(str(value), normal_style)

        # Convert column names to Paragraphs
        headers = [to_paragraph(col) for col in dataframe.columns]

        # Convert all data to Paragraphs
        data = [[to_paragraph(cell) for cell in row] for row in dataframe.values.tolist()]

        # Combine headers and data
        table_data = [headers] + data

        table = LongTable(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.color_scheme['table_header_bg']),
            ('TEXTCOLOR', (0, 0), (-1, 0), self.color_scheme['table_header_text']),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), self.color_scheme['table_body_bg']),
            ('TEXTCOLOR', (0, 1), (-1, -1), self.color_scheme['table_body_text']),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, self.color_scheme['table_grid'])
        ]))
        self.elements.append(table)
        self.elements.append(Spacer(1, 12))

    def add_explanation(self, text, icon_path=None, icon_width=24, icon_height=24, icon_position='left'):
        if 'Explanation' not in self.styles:
            self.add_explanation_style()
        
        if icon_path:
            explanation = IconAndText(icon_path, text, icon_width, icon_height, self.styles['Explanation'], icon_position)
        else:
            explanation = Paragraph(text, self.styles['Explanation'])
        
        self.elements.append(KeepTogether([explanation, CondPageBreak(20)]))
        self.elements.append(Spacer(1, 12))

    def add_ai_advice(self, advice_text):
        paragraphs = advice_text.split('\n')
        for para in paragraphs:
            if para.strip():
                if para.startswith('•') or para.startswith('-'):  # Bullet points
                    style = self.styles['Normal']
                    style.leftIndent = 20
                    p = Paragraph(para, style)
                elif para[0].isdigit() and para[1] == '.':  # Numbered lists
                    style = self.styles['Normal']
                    style.leftIndent = 20
                    p = Paragraph(para, style)
                else:  # Regular paragraphs
                    p = Paragraph(para, self.styles['Normal'])
                self.elements.append(p)
                self.elements.append(Spacer(1, 6))

    def add_markdown(self, markdown_text):
        lines = markdown_text.split('\n')
        for line in lines:
            if line.strip().startswith(('#', '##', '###')):  # Headers
                level = line.count('#')
                text = line.strip('#').strip()
                self.elements.append(Paragraph(text, self.styles[f'Heading{level}']))
            elif line.strip().startswith(('-', '*', '+')):  # List items
                indent = len(line) - len(line.lstrip())
                text = line.strip()[1:].strip()  # Remove the list marker and leading/trailing whitespace
                text = self.format_bold(text)  # Apply bold formatting
                bullet_style = ParagraphStyle('BulletList', parent=self.styles['BulletList'], leftIndent=20 + indent)
                self.elements.append(Paragraph(f'• {text}', bullet_style))
            elif line.strip():  # Regular paragraphs
                text = self.format_bold(line)  # Apply bold formatting
                self.elements.append(Paragraph(text, self.styles['Normal']))
            else:  # Blank lines
                self.elements.append(Spacer(1, 6))

    def format_bold(self, text):
        # Replace **text** with <b>text</b> for bold formatting
        return re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

    def format_text(self, text):
        # Replace **text** with <b>text</b> for bold
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        # Replace *text* or _text_ with <i>text</i> for italics
        text = re.sub(r'(\*|_)(.*?)\1', r'<i>\2</i>', text)
        # Replace `text` with <code>text</code> for inline code
        text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
        return text

    def _process_element(self, element):
        for child in element:
            if child.tag == 'p':
                self.elements.append(Paragraph(self._get_element_text(child), self.styles['Normal']))
            elif child.tag in ('h1', 'h2', 'h3'):
                self.elements.append(Paragraph(self._get_element_text(child), self.styles[child.tag]))
            elif child.tag in ('ul', 'ol'):
                items = []
                for item in child:
                    if item.tag == 'li':
                        items.append(ListItem(Paragraph(self._get_element_text(item), self.styles['Normal'])))
                self.elements.append(ListFlowable(items, bulletType='bullet' if child.tag == 'ul' else '1', leftIndent=20))
            else:
                self._process_element(child)

            if child.tag in ('p', 'h1', 'h2', 'h3', 'ul', 'ol'):
                self.elements.append(Spacer(1, 6))

    def _get_element_text(self, element):
        return ''.join(element.itertext()).strip()

    def _header_footer(self, canvas, doc):
        canvas.saveState()
        w, h = doc.pagesize

        # Header
        canvas.setFont('Helvetica-Bold', 16)
        canvas.drawString(inch, h - 0.75*inch, "Illumio Report")

        # Logo
        if self.logo_path:
            img = Image(self.logo_path, width=1.5*inch, height=0.5*inch)
            canvas.drawImage(self.logo_path, w - 2.5*inch, h - 0.75*inch, width=1.5*inch, height=0.5*inch)

        # Header line
        canvas.setStrokeColor(self.color_scheme['table_grid'])
        canvas.line(inch, h - inch, w - inch, h - inch)

        # Custom header text
        if self.header_text:
            canvas.setFont('Helvetica', 9)
            canvas.drawString(inch, h - 1.25*inch, self.header_text)

        # Footer line
        canvas.line(inch, inch, w - inch, inch)

        # Footer
        if self.footer_text:
            canvas.setFont('Helvetica', 9)
            canvas.drawString(inch, 0.75*inch, self.footer_text)

        # Page number
        canvas.setFont('Helvetica', 9)
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.drawRightString(w - inch, 0.75*inch, text)

        canvas.restoreState()

    def save(self, filename):
        doc = SimpleDocTemplate(filename, pagesize=letter, 
                                leftMargin=0.5*inch, rightMargin=0.5*inch,
                                topMargin=1*inch, bottomMargin=1*inch)
        
        # Create frames for the content
        content_frame = Frame(
            doc.leftMargin, doc.bottomMargin,
            doc.width, doc.height,
            leftPadding=0, rightPadding=0, # Remove default padding
            id='normal'
        )

        # Create a PageTemplate with the header and footer function
        template = PageTemplate(id='test', frames=content_frame, onPage=self._header_footer)

        # Add the template to the document
        doc.addPageTemplates([template])

        # Build the document
        doc.build(self.elements)

    def add_graph(self, image_path, width=6*inch, height=4*inch):
        img = Image(image_path, width=width, height=height)
        self.elements.append(img)
        self.elements.append(Spacer(1, 12))

    def add_paragraph(self, text):
        style = self.styles['Normal']
        paragraph = Paragraph(text, style)
        self.elements.append(paragraph)
        self.elements.append(Spacer(1, 12))

    def add_plotly_figure(self, fig, width=500, height=300):
        img_buffer = io.BytesIO()
        fig.write_image(img_buffer, format="png", width=width, height=height)
        img_buffer.seek(0)
        img = Image(img_buffer)

        # Check if there's enough space on the current page
        if self.current_y + height > self.page_height - 2 * self.margin:
            self.elements.append(PageBreak())
            self.current_y = 0

        self.elements.append(img)
        self.current_y += height + 20  # Add some padding

    def add_custom_style(self, style_name, properties):
        """
        Add a custom style to the report generator.
        
        :param style_name: Name of the new style
        :param properties: Dictionary of style properties
        """

        alignment_map = {
            'LEFT': TA_LEFT,
            'CENTER': TA_CENTER,
            'RIGHT': TA_RIGHT
        }

        # Convert color if it's in our color scheme
        if 'textColor' in properties and isinstance(properties['textColor'], str):
            properties['textColor'] = self.color_scheme.get(properties['textColor'], properties['textColor'])

        # Map string alignment to ReportLab enum
        if 'alignment' in properties:
            properties['alignment'] = alignment_map.get(properties['alignment'].upper(), TA_LEFT)

        # Create the style
        new_style = ParagraphStyle(
            name=style_name,
            parent=self.styles[properties.get('parent', 'Normal')],
            **properties  # Add all properties to the new style
        )

        # Add the new style to the styles dictionary
        self.styles.add(new_style)

    def add_image(self, image_path, max_width=6*inch, max_height=None):
        """
        Add an image to the report, maintaining its aspect ratio.
        
        :param image_path: Path to the image file
        :param max_width: Maximum width of the image (default: 6 inches)
        :param max_height: Maximum height of the image (default: None, meaning no height limit)
        """
        # Use ImageReader to get image dimensions
        img_reader = ImageReader(image_path)
        img_width, img_height = img_reader.getSize()

        # Calculate the aspect ratio
        aspect_ratio = img_width / img_height

        # Calculate new dimensions
        new_width = min(max_width, self.page_width - 2*inch)  # Subtract 2 inches for margins
        new_height = new_width / aspect_ratio

        # If max_height is specified and new_height exceeds it, recalculate
        if max_height and new_height > max_height:
            new_height = max_height
            new_width = new_height * aspect_ratio

        # Create the Image object with the calculated dimensions
        img = Image(image_path, width=new_width, height=new_height)
        
        # Add the image to the elements list
        self.elements.append(img)
        self.elements.append(Spacer(1, 12))  # Add some space after the image