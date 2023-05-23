from os import listdir
from os.path import join

from researchjournal.runlikeascientisstcommons import *

subject_dirs = listdir(scores_path)

for subject_dir in subject_dirs:
    subject_path = join(scores_path, subject_dir)




