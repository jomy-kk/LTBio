from os.path import join

import mne, pyedflib

from researchjournal.runlikeascientisstcommons import dataset_edf_path

# open with mne
raw = mne.io.read_raw_edf(join(dataset_edf_path, '3B8D', 'scientisst_chest.edf'), preload=False)

# open with pyedflib
f = pyedflib.EdfReader(join(dataset_edf_path, '3B8D', 'scientisst_chest.edf'))
