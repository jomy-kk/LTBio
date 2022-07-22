class Frequency(float):

    def __init__(self, value:float):
        self.value = float(value)

    def __str__(self):
        return str(self.value) + ' Hz'

    def __eq__(self, other):
        if isinstance(other, float):
            return other == self.value
        elif isinstance(other, Frequency):
            return other.value == self.value

    def __float__(self):
        return self.value

    def __copy__(self):
        return Frequency(self.value)
