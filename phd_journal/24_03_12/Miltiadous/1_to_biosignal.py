from datetime import timedelta
from glob import glob
from os.path import join, exists, split

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.sources.Miltiadous import MiltiadousDataset

common_path = '/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/fixed_denoised'
#common_path = '/Volumes/MMIS-Saraiv/Datasets/KJPP/test'
out_common_path = '/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/denoised_biosignal'
# socio_demog = '/Volumes/MMIS-Saraiv/Datasets/KJPP/demographics.csv'  # For future, when the demographics file is updated
socio_demog = '/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/participants.tsv'
source = MiltiadousDataset(socio_demog)

# Get recursively all .set files in common_path
all_files = glob(join(common_path, '*.set'))

for F in all_files:
    filename = split(F)[-1]
    print(F)

    # Make Biosignal object
    x = EEG(F, source)

    # Structure its name
    short_patient_code = x.patient_code
    out_filename = short_patient_code
    out_filepath = join(out_common_path, out_filename + '.biosignal')

    # If the file does not exist yet, save it
    if not exists(out_filepath):
        print(f"Done {filename}.")
        x.save(out_filepath)
        # Take a sneak peek as well
        if x.duration >= timedelta(seconds=30):
            try:
                x["T5"][:x.initial_datetime+timedelta(seconds=30)].plot(show=False, save_to=join(out_common_path, out_filename + '.png'))
            except IndexError:
                x["T5"][:x.domain[0].end_datetime].plot(show=False, save_to=join(out_common_path, out_filename + '.png'))
        else:
            x["T5"].plot(show=False, save_to=join(out_common_path, out_filename + '.png'))
    # Delete the object to free memory
    del x

