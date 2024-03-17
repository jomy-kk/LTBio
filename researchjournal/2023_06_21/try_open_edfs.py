from os.path import join

import mne, pyedflib

from researchjournal.runlikeascientisstcommons import dataset_edf_path

# open with mne
#raw = mne.io.read_raw_edf(join(dataset_edf_path, '93JD', 'empatica.edf'), preload=False)

# open with pyedflib
f = pyedflib.EdfReader(join(dataset_edf_path, '93JD', 'scientisst_chest.edf'))
f.close()
