
from os import mkdir
from os.path import join, isfile, isdir
from src.ltbio.biosignals.modalities.MultimodalBiosignal import MultimodalBiosignal
from researchjournal.runlikeascientisstcommons import *
from core.serializations.edf import load_from_edf

x = load_from_edf(join(dataset_edf_path, 'JD3K', 'scientisst_chest.edf'))

