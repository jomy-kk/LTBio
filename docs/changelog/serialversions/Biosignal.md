# Biosignal

## Serial Version 1

_Date Created: 01-06-2022_

```
(SERIALVERSION, name, source, patient, acquisition_location, associated_events, timeseries)
```

* `SERIALVERSION` equals 1.
* `name` is a `str` with the value of the biosignal's `__name` attribute.
* `source` is a `BiosignalSource` class, or the state of a `BiosignalSource` object, based on the value of the biosignal's `__source` attribute.
* `patient` is the state of the `Patient` referenced in the biosignal's `__patient` attribute.
* `acquisition_location` is a `BodyLocation` with the value of the biosignal's `__acquisition_location` attribute.
* `associated_events` is a tuple of the states of all `Event`s' referenced in the biosignal's `__associated_events` attribute.
* `timeseries` is a dictionary of the states of all `Timeseries`s' referenced in the biosignal's `__timeseries` attribute.

