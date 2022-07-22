# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: SupervisedTrainReport
# Description: Class SupervisedTrainReport, produces a PDF report for a SupervisingTrainer.

# Contributors: João Saraiva
# Created: 6/05/2022

# ===================================

from datetime import datetime

from fpdf.fpdf import FPDF

from ml.models.SupervisedModel import SupervisedModel


class SupervisedTrainReport(FPDF):
    def __init__(self):
        super().__init__()

        # Page dimensions
        self.MARGINS = 18
        self.PAGE_WIDTH = 210 - self.MARGINS*2
        self.PAGE_HEIGHT = 297

        # Full width picture dimensions
        self.FULL_PIC_WIDTH = self.PAGE_WIDTH
        self.FULL_PIC_HEIGHT = 0  # zero means whatever the image height is

        # Small picture dimensions
        self.SMALL_PIC_SEP = 6
        self.SMALL_PIC_WIDTH = self.PAGE_WIDTH/2 - self.SMALL_PIC_SEP/2
        self.SMALL_PIC_HEIGHT = self.SMALL_PIC_WIDTH

        self.set_margins(self.MARGINS, self.MARGINS, self.MARGINS)

    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(123, 3, 'Supervised Training Report', 0, 0, 'L')
        self.set_font('Arial', '', 10)
        #self.cell(25, 3)
        current_date = datetime.now().strftime("%d-%m-%Y")
        current_time = datetime.now().strftime("%H:%M:%S")
        self.multi_cell(50, 5, 'Date: {0}\nHour: {1}\nEngineer: João Saraiva'.format(current_date, current_time), align='R')

        self.ln(10)

    def print_model_description(self, model:SupervisedModel, **descriptors):
        self.__add_title_cell('MODEL DESCRIPTION')

        self.set_font('Arial', '', 10)
        self.cell(100, 5, 'Name: {0}'.format(model.name), 0, 0, 'L')
        self.__break_line()
        self.cell(100, 5, 'Design class: {0}'.format(type(model.design).__name__), 0, 0, 'L')

        self.ln(7)
        descriptions = "\t".join(['{0}={1}'.format(label, descriptors[label]) for label in descriptors])
        self.multi_cell(self.PAGE_WIDTH, 5, descriptions)

    def __break_line(self):
        self.ln(5)

    def __add_title_cell(self, text:str):
        self.ln(12)
        self.set_font('Arial', 'B', 10)
        self.cell(self.PAGE_WIDTH, 5, text, 0, 0, 'L')
        self.set_font('Arial', '', 10)
        self.ln(7)

    def __add_log_cell(self, text:str):
        self.__break_line()
        self.set_font('Arial', '', 10)
        self.set_fill_color(247, 247, 247)
        self.set_text_color(60, 60, 60)
        self.cell(self.PAGE_WIDTH, 5, '{0} '.format(datetime.now().strftime("%H:%M:%S")) + text, 0, 0, 'L', fill=True)
        self.set_text_color(0, 0, 0)
        self.x = self.l_margin


    def print_successful_instantiation(self):
        self.add_page()
        self.__add_log_cell('SupervisingTrainer instantiated successfully.')

    def print_start_of_train(self, current_train, total_trains, train_conditions):
        self.__add_title_cell("DESCRIPTION OF EXPERIMENT {}".format(current_train))
        self.multi_cell(self.PAGE_WIDTH, 5, str(train_conditions))
        self.__add_log_cell('Train {0} (of {1}) started.'.format(current_train, total_trains))

    def print_textual_results(self, **results):
        self.__add_title_cell("RESULTS")
        for label in results:
            self.cell(self.PAGE_WIDTH/4, 5, '{0} = {1}'.format(label, results[label]))

    def print_end_of_trains(self, total_trains):
        self.__add_log_cell('All ({0}) train-test sessions were completed successfully.'.format(total_trains))

    def print_loss_plot(self, image_path:str):
        self.__break_line()
        self.image(image_path, w=self.FULL_PIC_WIDTH, h=self.FULL_PIC_HEIGHT)

    def print_small_plots(self, image_paths:str):
        """
        Prints a grid of n lines and 2 columns.
        """
        self.__break_line()
        for i, image_path in enumerate(image_paths):
            if i%2 == 0:
                self.image(image_path, w=self.SMALL_PIC_WIDTH, h=self.SMALL_PIC_HEIGHT)
            else:
                self.image(image_path, w=self.SMALL_PIC_WIDTH, h=self.SMALL_PIC_HEIGHT,
                           x=self.x + self.SMALL_PIC_WIDTH + self.SMALL_PIC_SEP, y=self.y - self.SMALL_PIC_HEIGHT)


    def footer(self):
        # Page numbers in the footer
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')
