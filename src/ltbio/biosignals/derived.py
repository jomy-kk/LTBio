# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: src/ltbio/biosignals 
# Module: pseudo
# Description: 

# Contributors: João Saraiva
# Created: 07/03/2023

# ===================================

class DerivedBiosignal(Biosignal):
    """
    A DerivedBiosignal is a set of Timeseries of some extracted feature from an original Biosignal.
    It is such a feature that it is useful to manipulate it as any other Biosignal.
    """

    def __init__(self, timeseries, source = None, patient = None, acquisition_location = None, name = None, original: Biosignal = None):
        if original is not None:
            super().__init__(timeseries, original.source, original._Biosignal__patient, original.acquisition_location, original.name)
        else:
            super().__init__(timeseries, source, patient, acquisition_location, name)

        self.original = original  # Save reference


class ACCMAG(DerivedBiosignal):

    DEFAULT_UNIT = G(Multiplier._)

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None, original: ACC | None = None):
        super().__init__(timeseries, source, patient, acquisition_location, name, original)

    @classmethod
    def fromACC(cls):
        pass

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass

class RRI(DerivedBiosignal):

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None, original: ECG | None = None):
        super().__init__(timeseries, source, patient, acquisition_location, name, original)

    @classmethod
    def fromECG(cls):
        pass

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass


class HR(DerivedBiosignal):

    DEFAULT_UNIT = BeatsPerMinute()

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None, original: RRI | IBI | ECG | PPG | None = None):
        super(HR, self).__init__(timeseries, source, patient, acquisition_location, name, original)

    @classmethod
    def fromRRI(cls):
        pass

    @classmethod
    def fromIBI(cls):
        pass

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass

    def acceptable_quality(self):  # -> Timeline
        """
        Acceptable physiological values
        """
        return self.when(lambda x: 40 <= x <= 200)  # between 40-200 bpm


class IBI(DerivedBiosignal):

    DEFAULT_UNIT = Second()

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None, original: PPG | None = None):
        super().__init__(timeseries, source, patient, acquisition_location, name, original)

    @classmethod
    def fromPPG(cls):
        pass

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass

