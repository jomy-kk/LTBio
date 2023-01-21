




### Operations

#### Concatenation

`biosignal3 = biosignal1 + biosignal2` concatenates the beginning of all channels of `biosignal2` to the end of `biosignal1`, and stores it in `biosignal3`.

`biosignal1 += biosignal2` concatenates the beginning of all channels of `biosignal2` to the end of `biosignal1`, and stores it in `biosignal1`.

> Raises **Type Error** when the biosignals are from different modalities.

> Raises **Arithmetic Error** when the biosignals have a different number of channels or different channel names.

> Raises **Arithmetic Error** when the biosignals have associated patients with different codes.

> Raises **Arithmetic Error** when the biosignals have different associated acquisition locations.

> Raises **Arithmetic Error** when any channel of `biosignal2` comes before (in time) than any channel of `biosignal1`.



