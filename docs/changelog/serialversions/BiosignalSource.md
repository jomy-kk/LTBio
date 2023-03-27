# __BiosignalSource

`__BiosignalSource` is usually not instantiated as an object, so there are no states to serialize.
However, there are some sources that are instantiated, e.g., `Sense`, `Bitalino`. In these cases, the following serial versions apply.

## Serial Version 1

_Date Created: 01-06-2022_

```
(SERIALVERSION, others)
```

* `SERIALVERSION` equals 1.
* `others` is a dictionary of properties an instantiated `__BiosignalSource` object may have. 

