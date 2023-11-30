Biosignal File Format
=============

Below you can check the current and past versions of the .biosignal files (first row), and the Python library release where they were first introduced (last row).

+------------------------------+------------+------------+------------+---------------------+
| .biosignal                   | 2022.0     | 2022.1     | 2022.2     | 2023.0  (Newest)    |
+------------------------------+------------+------------+------------+---------------------+
| :code:`Biosignal`            | 1          | 1          | 1          | **2**               |
+------------------------------+------------+------------+------------+---------------------+
| :code:`__BiosignalSource`      | 1          | 1          | 1          | 1                   |
+------------------------------+------------+------------+------------+---------------------+
| :code:`Timeseries`           | 1          | **2**      | 2          | 2                   |
+------------------------------+------------+------------+------------+---------------------+
| :code:`Segment`              | 1          | 1          | 1          | **2**               |
+------------------------------+------------+------------+------------+---------------------+
| :code:`Unit`                 | 1          | 1          | 1          | 1                   |
+------------------------------+------------+------------+------------+---------------------+
| :code:`Event`                | 1          | 1          | 1          | 1                   |
+------------------------------+------------+------------+------------+---------------------+
| :code:`Patient`              | 1          | 1          | 1          | 1                   |
+------------------------------+------------+------------+------------+---------------------+
| :code:`MedicalCondition`     | 1          | 1          | **2**      | 2                   |
+------------------------------+------------+------------+------------+---------------------+
| :code:`SurgicalProcedure`    | 1          | 1          | 1          | 1                   |
+------------------------------+------------+------------+------------+---------------------+
| :code:`Medication`           | 1          | 1          | 1          | 1                   |
+------------------------------+------------+------------+------------+---------------------+
| Since Release                | 1.0.0      | 1.0.1      | 1.0.2      | 1.1.0               |
+------------------------------+------------+------------+------------+---------------------+



Any Biosignal and associated objects are stateful, so that they can be serialized. To that end, the states of Biosignal and associated objects are stored in Python tuples, with a pre-defined structure, before serialization with pickle. The content and ordering of this pre-defined structure has changed over time, giving rise to multiple serial versions of each class (middle rows). The changelogs of this structure for each class are given below, however this information might be irrelevant for the general user:

.. toctree::
   :maxdepth: 1

   Biosignal
   __BiosignalSource

.. tip::
   How this structure is created can be inspected in more detail in the methods :code:`__getstate__` and :code:`__setstate__` of each of these class.
