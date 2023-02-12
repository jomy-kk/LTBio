# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: src/ltbio/biosignals/timeseries 
# Module: Timeline
# Description: 

# Contributors: João Saraiva
# Created: 08/02/2023

# ===================================
from datetime import datetime, timedelta
from typing import Sequence

import matplotlib.pyplot as plt
from datetimerange import DateTimeRange
from matplotlib import cm
from matplotlib.dates import date2num
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


class Timeline():

    class Group():

        def __init__(self, intervals: Sequence[DateTimeRange] = [], points: Sequence[datetime] = [], name: str = None, color_hex: str = None):
            self.intervals = list(intervals)
            self.points = list(points)
            self.name = name
            self.color_hex = color_hex

        def __repr__(self):
            res = ''
            if len(self.intervals) > 1:
                res += ' U '.join(['[' + str(interval) + '[' for interval in self.intervals])
            if len(self.points) > 1:
                res += '\nand the following tiempoints:\n'
                res += ', '.join(['[' + str(point) + '[' for point in self.points])
            return res

        @property
        def initial_datetime(self) -> datetime:
            return min([interval.start_datetime for interval in self.intervals] + self.points)

        @property
        def final_datetime(self) -> datetime:
            return max([interval.end_datetime for interval in self.intervals] + self.points)

        @property
        def duration(self) -> timedelta:
            return sum([interval.timedelta for interval in self.intervals], timedelta())

        @property
        def has_only_intervals(self) -> bool:
            return len(self.intervals) > 0 and len(self.points) == 0

        @property
        def has_only_points(self) -> bool:
            return len(self.intervals) == 0 and len(self.points) > 0

        def _as_index(self) -> tuple:
            if self.has_only_intervals:
                return tuple([slice(interval.start_datetime, interval.end_datetime) for interval in self.intervals])
            if self.has_only_points:
                return tuple(self.points)
            return None

    def __init__(self, *groups: Group, name: str = None):
        self.groups = list(groups)
        self.__name = name

    @property
    def name(self):
        return self.__name if self.__name else "No Name"

    @name.setter
    def name(self, name: str):
        self.__name = name

    def __repr__(self):
        if len(self.groups) == 1:
            return repr(self.groups[0])
        else:
            res = ''
            for g in self.groups:
                res += f'\nGroup {g}\n'
                res += repr(g)
            return res

    def __and__(self, other):
        if isinstance(other, Timeline):
            groups = []
            groups += self.groups
            groups += other.groups
            group_names = [g.name for g in groups]
            if len(set(group_names)) != len(group_names):
                raise NameError('Cannot join Timelines with groups with the same names.')
            return Timeline(*groups, name = self.name + " and " + other.name)

    @property
    def initial_datetime(self) -> datetime:
        return min([g.initial_datetime for g in self.groups])

    @property
    def final_datetime(self) -> datetime:
        return max([g.final_datetime for g in self.groups])

    @property
    def duration(self) -> timedelta:
        if len(self.groups) == 1:
            return self.groups[0].duration
        else:
            return NotImplementedError()

    @property
    def is_index(self) -> bool:
        """
        Returns whether or not this can serve as an index to a Biosignal.
        A Timeline can be an index when:
        - It only contains one interval or a union of intervals (serves as a subdomain)
        - It only contains one point or a set of points (serves as set of objects)
        """
        return len(self.groups) == 1 and (self.groups[0].has_only_intervals ^ self.groups[0].has_only_points)

    def _as_index(self) -> tuple | None:
        if self.is_index:
            return self.groups[0]._as_index()

    def plot(self, show:bool=True, save_to:str=None):
        fig = plt.figure(figsize=(len(self.groups)*10, len(self.groups)*2))
        ax = plt.gca()
        legend_elements = []

        cmap = cm.get_cmap('tab20b')
        for y, g in enumerate(self.groups):
            color = g.color_hex
            if color is None:
                color = cmap(y/len(self.groups))

            for interval in g.intervals:
                start = date2num(interval.start_datetime)
                end = date2num(interval.end_datetime)
                rect = Rectangle((start, y + 0.4), end - start, 0.4, facecolor=color, alpha=0.5)
                ax.add_patch(rect)

            for point in g.points:
                ax.scatter(date2num(point), y + 0.95, color=color, alpha=0.5, marker = 'o', markersize=10)

            if len(self.groups) > 1:
                legend_elements.append(Line2D([0], [0], marker='o', color=color, label=g.name, markerfacecolor='g', markersize=10))

        ax.set_xlim(date2num(self.initial_datetime), date2num(self.final_datetime))
        ax.set_ylim(0, len(self.groups))
        ax.get_yaxis().set_visible(False)
        for pos in ['right', 'top', 'left']:
            plt.gca().spines[pos].set_visible(False)
        ax.xaxis_date()
        fig.autofmt_xdate()

        if len(self.groups) > 1:
            ax.legend(handles=legend_elements, loc='center')

        if self.name:
            fig.suptitle(self.name, fontsize=11)
        fig.tight_layout()
        if save_to is not None:
            fig.savefig(save_to)
        plt.show() if show else plt.close()

    def _repr_png_(self):
        self.plot()
