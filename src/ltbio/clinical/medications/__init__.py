from ..medications.Medication import Medication


class LEV(Medication):
    def __init__(self, dose=None, unit=None, frequency=None):
        super().__init__(dose, unit, frequency)

    @property
    def name(self):
        return 'Levetiracetam'


class CBZ(Medication):
    def __init__(self, dose=None, unit=None, frequency=None):
        super().__init__(dose, unit, frequency)

    @property
    def name(self):
        return 'Carbamazepine'


class ZNS(Medication):
    def __init__(self, dose=None, unit=None, frequency=None):
        super().__init__(dose, unit, frequency)

    @property
    def name(self):
        return 'Zonisamide'


class LTG(Medication):
    def __init__(self, dose=None, unit=None, frequency=None):
        super().__init__(dose, unit, frequency)

    @property
    def name(self):
        return 'Lamotrigine'


class PER(Medication):
    def __init__(self, dose=None, unit=None, frequency=None):
        super().__init__(dose, unit, frequency)

    @property
    def name(self):
        return 'Perampanel'


class Clobazam(Medication):
    def __init__(self, dose=None, unit=None, frequency=None):
        super().__init__(dose, unit, frequency)

    @property
    def name(self):
        return 'Clobazam'


class DZP(Medication):
    def __init__(self, dose=None, unit=None, frequency=None):
        super().__init__(dose, unit, frequency)

    @property
    def name(self):
        return 'Diazepam'


class VPA(Medication):
    def __init__(self, dose=None, unit=None, frequency=None):
        super().__init__(dose, unit, frequency)

    @property
    def name(self):
        return 'Valproic Acid'

class VIM(Medication):
    def __init__(self, dose=None, unit=None, frequency=None):
        super().__init__(dose, unit, frequency)

    @property
    def name(self):
        return 'Vimpat'


class ESL(Medication):
    def __init__(self, dose=None, unit=None, frequency=None):
        super().__init__(dose, unit, frequency)

    @property
    def name(self):
        return 'Eslicarbazepine acetate'


class CNZ(Medication):
    def __init__(self, dose=None, unit=None, frequency=None):
        super().__init__(dose, unit, frequency)

    @property
    def name(self):
        return 'Clonazepam'


class PHT(Medication):
    def __init__(self, dose=None, unit=None, frequency=None):
        super().__init__(dose, unit, frequency)

    @property
    def name(self):
        return 'Phenytoin'


class OXC(Medication):
    def __init__(self, dose=None, unit=None, frequency=None):
        super().__init__(dose, unit, frequency)

    @property
    def name(self):
        return 'Oxcarbazepine'


class LCS(Medication):
    def __init__(self, dose=None, unit=None, frequency=None):
        super().__init__(dose, unit, frequency)

    @property
    def name(self):
        return 'Lacosamide'


class TPM(Medication):
    def __init__(self, dose=None, unit=None, frequency=None):
        super().__init__(dose, unit, frequency)

    @property
    def name(self):
        return 'Topiramate'

class FEL(Medication):
    def __init__(self, dose=None, unit=None, frequency=None):
        super().__init__(dose, unit, frequency)

    @property
    def name(self):
        return 'Felbamate'

class GAB(Medication):
    def __init__(self, dose=None, unit=None, frequency=None):
        super().__init__(dose, unit, frequency)

    @property
    def name(self):
        return 'Gabapentin'
