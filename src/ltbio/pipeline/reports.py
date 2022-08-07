# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: pipeline
# Module: reports
# Description: 

# Contributors: João Saraiva
# Created: 06/08/2022

# ===================================
from abc import ABC, abstractmethod
from datetime import datetime

from fpdf.fpdf import FPDF


class PDFWriter(FPDF):
    def __init__(self):
        super().__init__()

        # Page dimensions
        self.MARGINS = 18
        self.PAGE_WIDTH = 210 - self.MARGINS * 2
        self.PAGE_HEIGHT = 297

        # Full width picture dimensions
        self.FULL_PIC_WIDTH = self.PAGE_WIDTH
        self.FULL_PIC_HEIGHT = 0  # zero means whatever the image height is

        # Small picture dimensions
        self.SMALL_PIC_SEP = 6
        self.SMALL_PIC_WIDTH = self.PAGE_WIDTH / 2 - self.SMALL_PIC_SEP / 2
        self.SMALL_PIC_HEIGHT = self.SMALL_PIC_WIDTH

        self.set_margins(self.MARGINS, self.MARGINS, self.MARGINS)

    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(123, 3, self.title, 0, 0, 'L')
        self.set_font('Arial', '', 10)
        # self.cell(25, 3)
        current_date = datetime.now().strftime("%d-%m-%Y")
        current_time = datetime.now().strftime("%H:%M:%S")
        self.multi_cell(50, 5, 'Date: {0}\nHour: {1}\nEngineer: João Saraiva'.format(current_date, current_time), align='R')

        self.ln(10)

    def footer(self):
        # Page numbers in the footer
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

    def __break_line(self):
        self.ln(5)

    # Addition of cells on demand:

    def add_section_cell(self, name: str):
        self.ln(15)
        self.set_font('Arial', 'B', 12)
        self.cell(self.PAGE_WIDTH, 5, name, 0, 0, 'C')
        self.set_font('Arial', '', 10)
        self.ln(12)

    def add_subsection_cell(self, text: str):
        self.ln(12)
        self.set_font('Arial', 'B', 10)
        self.cell(self.PAGE_WIDTH, 5, text.upper(), 0, 0, 'L')
        self.set_font('Arial', '', 10)
        self.ln(7)

    def add_log_cell(self, text: str):
        self.__break_line()
        self.set_font('Arial', '', 10)
        self.set_fill_color(247, 247, 247)
        self.set_text_color(60, 60, 60)
        self.cell(self.PAGE_WIDTH, 5, '{0} '.format(datetime.now().strftime("%H:%M:%S")) + text, 0, 0, 'L', fill=True)
        self.set_text_color(0, 0, 0)
        self.x = self.l_margin

    def add_text_cell(self, text: str):
        self.multi_cell(self.PAGE_WIDTH, 5, str(text))

    def add_image_fullwidth_cell(self, filepath: str):
        self.__break_line()
        self.image(filepath, w=self.FULL_PIC_WIDTH, h=self.FULL_PIC_HEIGHT)

    def add_image_grid_cell(self, filepaths: tuple[str]):
        self.__break_line()
        for i, image_path in enumerate(filepaths):
            if i % 2 == 0:
                self.image(image_path, w=self.SMALL_PIC_WIDTH, h=self.SMALL_PIC_HEIGHT)
            else:
                self.image(image_path, w=self.SMALL_PIC_WIDTH, h=self.SMALL_PIC_HEIGHT,
                                  x=self.x + self.SMALL_PIC_WIDTH + self.SMALL_PIC_SEP,
                                  y=self.y - self.SMALL_PIC_HEIGHT)


class Reporter(ABC):

    def __init__(self, writer: PDFWriter = None):
        if writer is not None:
            self.writer = writer
        else:
            self.writer = PDFWriter()

    @abstractmethod
    def body(self):
        pass

    def set_title(self, title: str):
        self.writer.title = title

    def begin_section(self, name: str):
        self.writer.add_section_cell(name)

    def begin_subsection(self, name: str):
        self.writer.add_subsection_cell(name)

    def add_text_block(self, text: str):
        self.writer.add_text_cell(text)

    def add_log_block(self, text: str):
        self.writer.add_log_cell(text)

    def add_image_fullwidth(self, filepath: str):
        self.writer.add_image_fullwidth_cell(filepath)

    def add_image_grid(self, filepaths: tuple[str]):
        self.writer.add_image_grid_cell(filepaths)

    def output_report(self, title: str, filepath: str):
        self.set_title(title)
        self.writer.add_page()
        self.body()  # write body
        self.writer.output(filepath)
