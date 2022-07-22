# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: clinical
# Module: CarpalTunnelRelease
# Description: Class CarpalTunnelRelease, to describe carpal tunnel surgeries.

# Contributors: João Saraiva
# Created: 09/07/2022

# ===================================

from datetime import datetime

from clinical.procedures.SurgicalProcedure import SurgicalProcedure


class CarpalTunnelRelease(SurgicalProcedure):

    def __init__(self, date: datetime = None, outcome=bool):
        super().__init__(date, outcome)

    @property
    def name(self):
        return "Carpal Tunnel Release"

    def __str__(self):
        outcome = ""
        if self.outcome is not None:
            outcome = "-- Successful outcome" if self.outcome else "-- Unsuccessful outcome"
        if self.date is None:
            return "{} {}".format(self.name, outcome)
        else:
            return "{} in {} {}".format(self.name, self.date, outcome)








