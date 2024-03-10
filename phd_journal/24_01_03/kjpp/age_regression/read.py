import pandas as pd


def get_insight_ages() -> dict[int, int]:
    insight_ages = {}
    df = pd.read_csv('/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/SocioDemog.csv', index_col=0)
    for i, row in df.iterrows():
        insight_ages[i] = row['AGE']
    return insight_ages


def get_kjpp_ages() -> dict[str, int]:
        df = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/KJPP/metadata_as_given.csv', sep=';')
        # Return {'EEG_GUID' : 'AgeMonthEEG'}
        return {row['EEG_GUID']: row['AgeMonthEEG']/12 for _, row in df.iterrows()}


def get_kjpp_sex() -> dict[str, int]:
    df = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/KJPP/metadata_as_given.csv', sep=';')
    # Return {'EEG_GUID' : 'AgeMonthEEG'}
    return {row['EEG_GUID']: 1 if row['Gender'] == 'Male' else 0 for _, row in df.iterrows()}


def get_insight_genders() -> dict[int, str]:
    insight_genders = {}
    df = pd.read_csv('/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/SocioDemog.csv', index_col=0, dtype=str)
    for i, row in df.iterrows():
        insight_genders[i] = row['SEX']
    return insight_genders


def get_insight_csf_score(parameter: str) -> dict[int, float]:
    insight_csf_score = {}
    df = pd.read_csv('/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/csf_m0.csv', index_col=0)
    for i, row in df.iterrows():
        insight_csf_score[i] = row[parameter]
    return insight_csf_score


def get_insight_pet_score() -> dict[int, float]:
    insight_csf_score = {}
    df = pd.read_csv('/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/pet_amyloid_m0.csv', index_col=0)
    for i, row in df.iterrows():
        insight_csf_score[i] = row['SUVR GLOBAL']
    return insight_csf_score


