import psutil


def activityName(name):
    if 'run' in name or 'sprint' in name:
        return 'Run'
    if 'lift' in name:
        return 'Lift'
    if 'gesticulate' in name:
        return 'Gesticulate'
    if 'walk_before' in name or 'walk-before' in name or 'downstairs' in name or 'elevator_walk' in name or 'stairs' in name or 'walk_indoors' in name:
        return 'Walk-Before'
    if 'walk_after' in name or 'walk-after' in name:
        return 'Walk-After'
    if 'baseline' in name or 'elevator_up' in name:
        return 'Baseline'
    if 'jumps' in name:
        return 'Jumps'
    if 'greetings' in name:
        return 'Greetings'
    return name



def exchangeChannels(code, event_name):
    return (code, event_name) in (
        ('3B8D', 'walk_after'),
        ('03FH', 'baseline'),
        ('3RFH', 'run'),
        ('3RFH', 'walk_after'),
        ('4JF9', 'run'),
        ('93DK', 'baseline'),
        ('93DK', 'greetings'),
        ('93DK', 'jumps'),
        ('93DK', 'walk_before'),
        ('93DK', 'run'),
        ('F408', 'run'),
        ('F408', 'walk_after'),
        ('H39D', 'walk_after'),
        ('LAS2', 'lift-1'),
        ('LAS2', 'elevator_walk'),
        ('LAS2', 'walk_before'),
        ('LK27', 'walk_after'),
        ('K2Q2', 'walk_before'),
        ('K2Q2', 'walk_after'),
        ('LDM5', 'walk_after'),
    )


def dismiss(code, event_name):
    return (code, event_name) in (
        ('3B8D', 'run'),
        ('F408', 'walk_before'),
        ('H39D', 'run-3'),
        ('H39D', 'run-4'),
        ('H39D', 'run-5'),
        ('H39D', 'sprint'),
        ('KF93', 'run'),
        ('LAS2', 'jumps'),
        ('LDM5', 'lift'),
        ('K2Q2', 'run'),
    )

def invert_manually_band(code):
    return code in ('93JD', 'LK27')

def invert_manually_gel(code):
    return code in ('LK27', )

codes_dates = (
    ('H39D', '07_08_2022'),

    ('H39D', '17_06_2022'),
    ('ME93', '20_06_2022'),
    ('LAS2', '21_06_2022'),
    ('93DK', '22_06_2022'),
    ('3RFH', '25_06_2022'),
    ('03FH', '26_06_2022'),
    ('F408', '28_06_2022'),
    ('KS03', '30_06_2022'),

    ('4JF9', '02_07_2022'),
    ('KF93', '04_07_2022'),
    ('LDM5', '06_07_2022'),
    ('JD3K', '07_07_2022'),
    ('AP3H', '13_07_2022'),
    ('3B8D', '14_07_2022'),
    ('93JD', '27_07_2022'),


    ('LK27', '11_08_2022'),
    ('K2Q2', '14_08_2022'),
)

codes_dates_smaller = (
    ('LK27', '11_08_2022'),
)

two_underscores_subjects = ('K2Q2', 'KS03', 'LAS2', 'LDM5', 'LK27', 'ME93')

activities = ('Baseline', 'Lift', 'Greetings', 'Gesticulate', 'Jumps', 'Walk-Before', 'Run', 'Walk-After')


common_path = "/Users/saraiva/Library/CloudStorage/OneDrive-UniversidadedeLisboa/11º e 12º Semestres/Tese/Aquisições"


def findDate(patientCode):
    for code, date in codes_dates:
        if code == patientCode:
            return date  # devolve o primeiro. Cuidado com o H39D


