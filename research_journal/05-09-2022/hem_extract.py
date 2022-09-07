

from ltbio.biosignals.sources.HEM import HEM
from ltbio.biosignals.modalities import ECG
from datetime import datetime


path_ = 'D:\\PreEpiSeizures\\Patients_HEM\\FCSFDM\\ficheiros'
options = {'date1': datetime(2021, 4, 15, 14, 45, 49),
           'date2': datetime(2021, 4, 15, 14, 49, 32)}
# options = {'date1': datetime(2021, 4, 15, 10, 45, 49),
#           'date2': datetime(2021, 4, 15, 11, 49, 32)}

data = ECG(path_, HEM)
