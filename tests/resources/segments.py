# -- encoding: utf-8 --
# ===================================
# ScientISST LTBio | Long-Term Biosignals
#
# Package: tests.resources
# Module: segments
# Description: Factory of Segment objects for testing purposes
#
# Contributors: João Saraiva
# Created: 17/05/2016
# Last Updated: 19/05/2016
# ===================================

import numpy as np

from ltbio.biosignals import Segment


# ARRAYS
# You can use this to assert the sample values according to length and group
small_samples_1 = np.array([506.0, 501.0, 497.0, 374.5, 383.4, 294.2])
small_samples_2 = np.array([502.0, 505.0, 505.0, 924.3, 293.4, 383.5])
small_samples_3 = np.array([527.0, 525.0, 525.0, 849.2, 519.5, 103.4])
medium_samples_1 = np.array([686.4, 753.6, 845.9, 806.1, 247.7, 107.1, 598.2, 518.8, 502.5, 641.6, 582.9, 139.1])
medium_samples_2 = np.array([412.3, 702.9, 731.2, 200.6, 517. , 428. , 298.9, 419.4, 289.5, 249.4, 880.1, 382.4])
medium_samples_3 = np.array([678.9, 707.8, 144.4, 908.3, 723.2, 912.2, 789.5, 428.1, 919.8, 876. , 333.3, 709.1])
large_samples_1 = np.array([ 49.5, 367. , 503.8, 111.5, 853.3, 503.1, 312. , 167.8, 417. ,
       413.7, 449.7, 829.8, 306.6, 169.5, 774.4, 845. , 777.6, 605.6,
       208.9, 364.4, 364.5, 872.8, 704.1, 625.2])
large_samples_2 = np.array([422.4, 989.6, 381. , 449.3, 231.6,  29.3, 753.9,  88. , 257.1,
       125.4, 666.5, 943.2, 900.4, 755.2, 857.1, 607.8,  97.8,  48. ,
        86.2, 582.2, 317.1, 546.2,  97.5, 403.9])
large_samples_3 = np.array([907.2, 787.4, 391.8, 505.4, 606. , 597.1, 957.9, 713.7, 957.7,
       151. , 725.3, 163.6, 882.9, 933.2,   3.9, 754.4, 892.5,  36.9,
       880.6, 139.6, 305.9, 508. , 618.6, 235.7])


def get_segment_length(length: str) -> int:
    if length == 'small':
        return 6
    if length == 'medium':
        return 12
    if length == 'large':
        return 24


def get_segment(length: str, group: int) -> Segment:
    """
    Use get_segment to get a new Segment object with samples for testing purposes.
    Samples were generated randomly with amplitude values between 0 and 1000.

    :param length: Length of the segment: 'small' = 6, 'medium' = 12, 'large' = 24.
    :param group: Group of examples: 1, 2 or 3.
    """
    if length == 'small':
        if group == 1:
            return Segment(small_samples_1)
        if group == 2:
            return Segment(small_samples_2)
        if group == 3:
            return Segment(small_samples_3)
    if length == 'medium':
        if group == 1:
            return Segment(medium_samples_1)
        if group == 2:
            return Segment(medium_samples_2)
        if group == 3:
            return Segment(medium_samples_3)
    if length == 'large':
        if group == 1:
            return Segment(large_samples_1)
        if group == 2:
            return Segment(large_samples_2)
        if group == 3:
            return Segment(large_samples_3)
