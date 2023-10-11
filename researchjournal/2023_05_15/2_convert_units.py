from datetime import datetime
from os.path import join, isfile

from ltbio.biosignals import Biosignal
from ltbio.biosignals.sources import Sense
from ltbio.biosignals.timeseries.Unit import Volt, Multiplier, Siemens, G
from researchjournal.runlikeascientisstcommons import *


for code in subject_codes:
    subject_path = join(dataset_biosignal_path, code)
    print(code)


    modality = 'ecg'
    path = join(dataset_biosignal_path, code, modality + biosignal_file_suffix)
    if isfile(path):
        print('\t', modality)
        x = Biosignal.load(path)
        x.convert(Volt(Multiplier.m))
        #x.preview.plot()
        pass
        x.save(path)

    modality = 'acc_chest'
    path = join(dataset_biosignal_path, code, modality + biosignal_file_suffix)
    if isfile(path):
        print('\t', modality)
        x = Biosignal.load(path)
        x.convert(G())
        #x.preview.plot()
        pass
        x.save(path)
    """

    modality = 'emg'
    path = join(dataset_biosignal_path, code, modality + biosignal_file_suffix)
    if isfile(path):
        print('\t', modality)
        x = Biosignal.load(path)
        x.convert(Volt(Multiplier.m))
        #x.preview.plot()
        pass
        x.save(path)

    """
    modality = 'eda'
    path = join(dataset_biosignal_path, code, modality + biosignal_file_suffix)
    if isfile(path):
        print('\t', modality)
        x = Biosignal.load(path)
        if 'dry' in x and 'gel' in x:
            dry = x['dry']
            gel = x['gel']
            gel._Biosignal__source = Sense
            gel.convert(Siemens(Multiplier.u))
            x = dry & gel
            x.name = gel.name
        elif 'gel' in x:
            x.convert(Siemens(Multiplier.u))
        #x.preview.plot()
        pass
        x.save(path)

    modality = 'acc_e4'
    path = join(dataset_biosignal_path, code, modality + biosignal_file_suffix)
    if isfile(path):
        print('\t', modality)
        x = Biosignal.load(path)
        name = str(x.name)
        x = x / 64
        x.name = name
        pass
        #x.preview.plot()
        x.save(path)


    #exit(0)
