# -- encoding: utf-8 --
from copy import deepcopy
# ===================================

# IT - LongTermBiosignals

# Package: src/ltbio/biosignals/timeseries 
# Module: Timeline
# Description: 

# Contributors: João Saraiva
# Created: 08/02/2023

# ===================================
from datetime import datetime, timedelta
from functools import reduce
from typing import Sequence, List

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
            if 0 < len(self.intervals):
                if len(self.intervals) < 10:
                    res += ' U '.join(['[' + str(interval) + '[' for interval in self.intervals])
                else:
                    res += f'{len(self.intervals)} intervals with {self.duration} of total duration'
            if 0 < len(self.points):
                if len(self.points) < 10:
                    res += '\nand the following timepoints:\n'
                    res += ', '.join(['[' + str(point) + '[' for point in self.points])
                else:
                    res += f'\nand {len(self.points)} timepoints.\n'
            return res

        @property
        def initial_datetime(self) -> datetime | None:
            all_datetimes = [interval.start_datetime for interval in self.intervals] + self.points
            if len(all_datetimes) > 0:
                return min([interval.start_datetime for interval in self.intervals] + self.points)
            else:
                return None

        @property
        def final_datetime(self) -> datetime | None:
            all_datetimes = [interval.end_datetime for interval in self.intervals] + self.points
            if len(all_datetimes) > 0:
                return max([interval.end_datetime for interval in self.intervals] + self.points)
            else:
                return None

        @property
        def duration(self) -> timedelta:
            return sum([interval.timedelta for interval in self.intervals], timedelta())

        @property
        def has_only_intervals(self) -> bool:
            return len(self.intervals) > 0 and len(self.points) == 0

        @property
        def has_intervals(self) -> bool:
            return len(self.intervals) > 0

        @property
        def has_only_points(self) -> bool:
            return len(self.intervals) == 0 and len(self.points) > 0

        @property
        def has_points(self) -> bool:
            return len(self.points) > 0

        @property
        def is_empty(self):
            return len(self.intervals) == 0 and len(self.points) == 0

        def _as_index(self) -> tuple:
            if self.has_only_intervals:
                return tuple(self.intervals)
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
                res += f'\nGroup {g.name}\n'
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

    def __getitem__(self, key):
        if isinstance(key, str):
            for g in self.groups:
                if g.name == key:
                    return g
        else:
            raise TypeError('Invalid argument type.')

    @property
    def group_names(self) -> set[str]:
        return set(g.name for g in self.groups)

    @property
    def initial_datetime(self) -> datetime:
        """
        Finds the minimum initial datetime of all groups.
        Careful: Some groups return None if they are empty.
        """
        return min([g.initial_datetime for g in self.groups if g.initial_datetime is not None])

    @property
    def final_datetime(self) -> datetime:
        """
        Finds the maximum final datetime of all groups.
        Careful: Some groups return None if they are empty.
        """
        return max([g.final_datetime for g in self.groups if g.final_datetime is not None])

    @property
    def has_single_group(self) -> bool:
        return len(self.groups) == 1

    @property
    def single_group(self) -> Group:
        return self.groups[0] if self.has_single_group else None

    @property
    def duration(self) -> timedelta:
        if len(self.groups) == 1:
            return self.groups[0].duration
        else:
            return NotImplementedError()

    @property
    def is_empty(self) -> bool:
        return all([g.is_empty for g in self.groups])

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

    def plot(self, show: bool = True, save_to: str = None):
        fig = plt.figure(figsize=(len(self.groups)*10, len(self.groups)*2))
        ax = plt.gca()
        legend_elements = []

        cmap = cm.get_cmap('tab20b')

        if not self.is_empty:
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

    @classmethod
    def union(cls, *timelines):
        # Check input
        if not all(isinstance(tl, Timeline) for tl in timelines):
            raise TypeError("Give objects Timeline to Timeline.union.")
        if len(timelines) < 2:
            raise ValueError("Give at least 2 Timelines to compute their union.")

        # Get sets of intervals of each Timeline
        tl_intervals = []
        for i, tl in enumerate(timelines):
            if tl.has_single_group and tl.single_group.has_only_intervals:
                tl_intervals.append(tl.single_group.intervals)
            else:
                raise AssertionError(f"The {i+1}th Timeline does not have a single group with only intervals.")

        # Binary function
        def union_of_two_timelines(intervals1: List[DateTimeRange], intervals2: List[DateTimeRange]):
            intervals = intervals1 + intervals2
            intervals.sort(key=lambda x: x.start_datetime)
            union = [intervals[0]]
            for i in range(1, len(intervals)):
                if union[-1].end_datetime >= intervals[i].start_datetime:
                    union[-1].set_end_datetime(max(union[-1].end_datetime, intervals[i].end_datetime))
                else:
                    union.append(intervals[i])
            return union

        res_intervals = reduce(union_of_two_timelines, tl_intervals)
        return Timeline(Timeline.Group(res_intervals), name=f"Union of " + ', '.join(tl.name for tl in timelines))

    @classmethod
    def intersection(cls, *timelines):
        # Check input
        if not all(isinstance(tl, Timeline) for tl in timelines):
            raise TypeError("Give objects Timeline to Timeline.intersection.")
        if len(timelines) < 2:
            raise ValueError("Give at least 2 Timelines to compute their intersection.")

        # Binary function
        def intersection_of_two_timelines(intervals1: List[DateTimeRange], intervals2: List[DateTimeRange]):
            intervals1.sort(key=lambda x: x.start_datetime)
            intervals2.sort(key=lambda x: x.start_datetime)

            intersection = []
            i, j = 0, 0
            while i < len(intervals1) and j < len(intervals2):
                if intervals1[i].end_datetime <= intervals2[j].start_datetime:
                    i += 1
                elif intervals2[j].end_datetime <= intervals1[i].start_datetime:
                    j += 1
                else:
                    start = max(intervals1[i].start_datetime, intervals2[j].start_datetime)
                    end = min(intervals1[i].end_datetime, intervals2[j].end_datetime)
                    intersection.append(DateTimeRange(start, end))
                    if intervals1[i].end_datetime <= intervals2[j].end_datetime:
                        i += 1
                    else:
                        j += 1

            return intersection

        # Get sets of intervals of each Timeline
        # Case A: all Timelines have a single group with only intervals
        if all(tl.has_single_group for tl in timelines):
            if any(tl.single_group.has_points for tl in timelines):
                raise NotImplementedError("Give Timelines with only intervals.")
            elif any(tl.single_group.has_intervals for tl in timelines):
                tl_intervals = []
                for i, tl in enumerate(timelines):
                        tl_intervals.append(tl.single_group.intervals)

                res_intervals = reduce(intersection_of_two_timelines, tl_intervals)
                return Timeline(Timeline.Group(res_intervals, name=tl.single_group.name),
                                name=f"Intersection of " + ', '.join(tl.name for tl in timelines))
            else:
                return Timeline(Timeline.Group(name=timelines[0].single_group.name),
                                name=f"Intersection of " + ', '.join(tl.name for tl in timelines))  # empty Timeline

        # Case B: all Timelines have the same number of groups with matching names, and each group has only intervals:
        else:
            group_names = timelines[0].group_names
            if all(tl.group_names == group_names for tl in timelines):
                if all([g.has_only_intervals for g in tl.groups] for tl in timelines):
                    intervals_by_group = {name: [] for name in group_names}
                    for i, tl in enumerate(timelines):
                        for g in tl.groups:
                            intervals_by_group[g.name].append(g.intervals)

                    for name in group_names:
                        intervals_by_group[name] = reduce(intersection_of_two_timelines, intervals_by_group[name])

                    return Timeline(*[Timeline.Group(intervals_by_group[name], name=name) for name in group_names],
                                    name=f"Intersection of " + ', '.join(tl.name for tl in timelines))
                else:
                    raise NotImplementedError("Give Timelines with only intervals.")

    def __sub__(self, other):
        """
        Returns A\B (A except B), where A and B are Timelines.
        """

        if not isinstance(other, Timeline):
            raise TypeError("The second value should be a Timeline.")

        def _binary_except(i1: DateTimeRange, i2: DateTimeRange) -> List[DateTimeRange]:
            return i1.subtract(i2)  # Returns a list of 0, 1 or 2 intervals

        def _set_except_set(intervals1: List[DateTimeRange], intervals2: List[DateTimeRange]):
            # Create a copy of intervals1
            new_intervals1 = deepcopy(intervals1)
            # For each interval in intervals2
            for i2 in intervals2:
                # Check if it intersects with any interval in new_intervals1
                for index, i1 in enumerate(new_intervals1):
                    if i1.is_intersection(i2):
                        # If yes, compute i1\i2 and substitute the result in new_intervals1, in the exact location where i1 was
                        new_intervals1.pop(index)
                        i1_except_i2 = _binary_except(i1, i2)
                        if len(i1_except_i2) == 1: # this is needed because there might be two intervals in i1_except_i2
                            new_intervals1.insert(index, i1_except_i2[0])
                        if len(i1_except_i2) == 2:
                            new_intervals1.insert(index, i1_except_i2[0])
                            new_intervals1.insert(index+1, i1_except_i2[1])

                        #break  # go to the next interval in intervals2, which its subtraction is now going to be computed based on the updated new_intervals1

            return new_intervals1  # return the last updated version of new_intervals1


        # Case A: both Timelines have a single group with only intervals
        if self.has_single_group and other.has_single_group:
            if self.single_group.has_points or other.single_group.has_points:
                raise NotImplementedError("Give Timelines with only intervals.")
            elif not self.single_group.has_intervals:
                return self  # A\B = A, when A is empty
            else:
                if not other.single_group.has_intervals:
                    return self  # A\B = A, when B is empty
                else:
                    intervals1 = self.single_group.intervals
                    intervals2 = other.single_group.intervals
                    res = _set_except_set(intervals1, intervals2)
                    return Timeline(Timeline.Group(res, name=self.single_group.name),
                                    name=f"{self.name} except {other.name}")

        # Case B: both Timelines have the same number of groups with matching names, and each group has only intervals:
        else:
            if self.group_names == other.group_names:
                if self.single_group.has_points or other.single_group.has_points:
                    raise NotImplementedError("Give Timelines with only intervals.")

                intervals_by_group = {name: [] for name in self.group_names}
                for g in self.groups:
                    intervals1 = g.intervals
                    intervals2 = other[g.name].intervals
                    res = _set_except_set(intervals1, intervals2)
                    intervals_by_group[g.name] = res

                return Timeline(*[Timeline.Group(intervals_by_group[g.name], name=g.name) for g in self.groups],
                                name=f"{self.name} except {other.name}")
            else:
                raise NotImplementedError("Give Timelines with matching group names.")

    def agglomerate(self, min_interval: timedelta, max_delta: timedelta):
        """
        Agglomerates the intervals of a Timeline, based on a minimum interval and a maximum delta.
        It can be thought as a filter or a smoother of time intervals.
        It's useful when the Timeline has a lot of small intervals, for instance, distancing milliseconds from each
        other, that would be rather relevant for analysis if agglomerated.
        :param min_interval: minimum interval duration every interval should have, inclusively
        :param max_delta: maximum duration between two consecutive intervals to be agglomerated, inclusively
        :return: a new Timeline with agglomerated intervals
        """

        def _agglomerate(intervals: List[DateTimeRange], min_interval: timedelta, max_delta: timedelta):
            if len(intervals) == 0:
                return intervals

            # Agglomerate every consecutive pair of intervals
            new_intervals = [intervals[0], ]
            for i in intervals[1:]:
                if i.start_datetime - new_intervals[-1].end_datetime <= max_delta:
                    new_intervals[-1] = DateTimeRange(new_intervals[-1].start_datetime, i.end_datetime)  # substitute
                else:
                    new_intervals.append(i)  # add

            # Remove any interval is smaller than min_interval
            new_intervals = [i for i in new_intervals if i.timedelta >= min_interval]
            return new_intervals

        # Repeat the agglomeration for every group
        new_groups = []
        for g in self.groups:
            new_groups.append(Timeline.Group(_agglomerate(g.intervals, min_interval, max_delta), name=g.name))

        # Return a new Timeline with the agglomerated groups
        return Timeline(*new_groups, name=f"{self.name} (agglomerated)")


    EXTENSION = '.timeline'

    def save(self, save_to: str):
        # Check extension
        if not save_to.endswith(Timeline.EXTENSION):
            save_to += Timeline.EXTENSION
        # Write
        from _pickle import dump
        with open(save_to, 'wb') as f:
            dump(self, f)

    @classmethod
    def load(cls, filepath: str):
        # Check extension
        if not filepath.endswith(Timeline.EXTENSION):
            raise IOError("Only .timeline files are allowed.")

        # Read
        from _pickle import load
        with open(filepath, 'rb') as f:
            timeline = load(f)
            return timeline
