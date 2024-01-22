from glob import glob
from os.path import join, exists

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.sources.INSIGHT import INSIGHT

common_path = '/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/autopreprocessed'
out_common_path = '/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/autopreprocessed_biosignal'
socio_demog = '/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/SocioDemog.csv'
source = INSIGHT(socio_demog)

# Get recursively all .edf files in common_path
all_edf_files = glob(join(common_path, '**/*.set'), recursive=True)

for filepath in all_edf_files:
    out_file_path = join(out_common_path, filepath.split('/')[-1] + '.biosignal')
    if not exists(out_file_path):
        print(filepath)
        x = EEG(filepath, source)
        x["T5"].plot(show=False, save_to=join(out_common_path, filepath.split('/')[-1].replace('.set', '.png')))
        x.save(out_file_path)
        del x

