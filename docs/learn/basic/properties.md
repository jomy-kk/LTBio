# Enrich your Biosignals

You have noticed by now that ``Biosignal`` objects gather all information about a biosignal in an holistic way, and not just the data
samples. But ``Biosignal`` objects will only hold the information you give or that their `BiosignalSource` deducted.

## Print to see me

You can see everything a ``Biosignal`` contains by simply printing it 📃:

For instance, ```print(biosignal)``` will give you:

```
Name: 4H9A Hospital vEEG
Type: ECG (mV)
Location: Chest
Number of Channels: 2
Channels: Chest Lead I, Chest Lead II
Sampling Frequency: 254 Hz
Useful Duration: 3 days, 4 hours and 27 minutes (76.45h)
Source: Hospital Egas Moniz
```

Cool, right? The more metadata the ``Biosignal`` has, the more details the print instruction will show.
So, let's explore in detail what you can associate to biosignals.

## Clinical Information

We're done will all the Excel sheets you have with your patients info. Instead, you can store that information inside `Biosignal` objects.
And the best part is that info is not just sitting there. LTBio has a lot of features to make the most of that information in semantic way
to your research. That will become clear later.

Now, let's see how you can associate clinical information with the ``clinical`` package.

### Patient
A `Patient` is a subject with a name, an age, a biological sex, a collection of medical conditions, of surgical procedures, and of
medications, and a set of textual notes. All these properties are optional. Additionally, a `Patient` has an alphanumeric code, which is a 
mandatory property. This code is important in situations where a `Patient` needs to be unequivocally identified, or in a cohort where each 
`Patient` must have a unique code.

For example, here is how you can instantiate a `Patient`:

```
p1 = Patient('001', 'John', 12, Sex.M)
```

When instantiating a `Biosignal`, if a `Patient` is associated, these properties can be accessed and be
useful throughout the biosignal processing and analysis. One could instante the `Biosignal` like this:

```
john_eeg = EEG('pathToFile', HSM, patient=p1)
```

So, we know our patient name, age, sex, and we can identify them. Although, age and sex can be useful for analysis, we often are interested in
much more, such as their medical conditions, medications, and past surgical procedures. These are also objects that can
be associated with `Patient`.

#### 🦠 Medical Conditions

A medical condition is any condition given by a medical diagnosis.
`MedicalCondition` is an abstract class, so that multiple medical conditions can be extended according to the
needs of each project. Its internal structure contains the years since diagnosis, and it should be extended in each subclass 
accordingly to which information is useful to maintain organised.

For instance, we can say John has epilepsy like this:

```
epilepsy = Epilepsy(years_since_diagnosis=6)
p1 = Patient('001', 'John', 12, Sex.M, conditions=[epilepsy, ])
```

#### 💊 Medications

A `Medication` is any prescribed therapeutic. It has a name (`string`), a dose (`float`), a unit (`Unit`), and a frequency (`string`).

Clonazepam is a standard anti-epileptic drug. We can say John is taking it regularly like this:
```
cnz = CNZ(dose=10, unit=Grams(Multiplier.m), frequency=’Everyday at 10 am.’)
p1 = Patient('001', 'John', 12, Sex.M, conditions=[epilepsy, ], medications=[cnz, ])
```

#### 💉 Surgical Procedures

A `SurgicalProcedure` describes a surgical procedure a patient has underwent and its outcome. It has a name, a date, a time and an outcome.

For instance, we can say John underwent a relsease of the carpal tunel like this:

```
carpal = CarpalTunnelRelease(datetime(2017, 4, 2), outcome=True)
p1 = Patient('001', 'John', 12, Sex.M, conditions=[epilepsy, ], medications=[cnz, ], procedures=[carpal, ])
```

## Body Locations

A `BodyLocation` is a region in the human body. These can range from common names like chest, wrist, or scalp, to more technical
standardised locations like where ECG electrodes are placed on the chest (V1, V2, V3, …), or where EEG electrodes are placed on the scalp
(F1, FP1, Cz, …).
There is a class in ``clinical`` termed ``BodyLocation``, comprehending an enumeration of human anatomical locations. You may add more if
necessary to your project.

There are multiple situations in which you'll have to use values of this enumeration. The two main ones are:

* Associating an acquisition location to a `Biosignal`: Let's say you have an ECG, acquired from a wearable in the abdomen and a EDA acquired
at the left wrist. You can codify this as:
```
ecg = ECG('pathToFile', HSM, acquisition_location = BodyLocation.ABDOMEN)
eda = EDA('pathToFile', HSM, acquisition_location = BodyLocation.WRIST_L)
```

* Naming each channel after were the electrode was placed. Let's say you have an EEG acquired with 19 electrodes on the scalp. You can name
the channels with the standardised 10-20 nomencalture, like this:
```
eeg = EEG({
    BodyLocation.FP1: Timeseries(...),
    BodyLocation.FP2: Timeseries(...),
    BodyLocation.CZ: Timeseries(...),
    BodyLocation.T3: Timeseries(...),
    ...
})
```

## Units of Measure

A `Unit` represents a unit of measure of some variable. The common units of biological variables are Volt (`Volt`), Siemens (`Siemens`), 
G force (`G`), Celsius degree (`DegreeCelsius`), decibels (`Decibels`), etc. There are many more implemented. You may add more if
necessary to your project. A `Unit` object can have a multiplier. Common multipliers are available in the `Multiplier` enumeration: nano, micro, milli, thousand, etc.

It is not mandotory you associate units to your Timeseries. But there are a couple of reasons why you might want to do it.
* To facilitate the conversion of `Timeseries` from one unit to another.
* To visualise the plots with the correct units and axis labels.

You can associate units when instantiating `Timeseries`, like this:

```
ts = Timeseries(samples, datetime(2023, 1, 6, 17, 23, 40), 256, Volt(Multiplier.m))
```


## Inspect All Properties

### Getters

You can **get** any of the following properties of a `Biosignal`:

* `name` returns the biosignal associated name, if any (in `string`).
* `channel_names` returns a set with the channel labels (in `string` or `BodyLocation`).
* `sampling_frequency` returns the the sampling frequency of every channel, if equal (in `float`).
* `acquisition_location` returns the body location where the biosignal was acquired (in `BodyLocation`).
* `source` returns the source where the Biosignal was acquired: hospital, device, etc. (in `BiosignalSource`).
* `patient_code` returns the code of the patient whose the biosignal belongs (in `int` or `string`).
* `type` returns the biosignal modality (in any `Biosignal` subclass).
* `initial_datetime` returns the initial datetime of the channel that starts the earliest (in `datetime`).
* `final_datetime` returns the final datetime of the channel that ends the latest (in `datetime`).
* `domain` returns the time intervals when samples were acquired (in `tuple` of `DateTimeRange`).
* `duration` returns the useful duration, not counting with interruptions (in `timedelta`).
* `events` returns all events annotated (see next Chapter).

To get the number of channels, use ``len``:
```
len(biosignal)
```

### Setters

You can reset the `name` like this: `biosignal.name = "New name"`.

You can reset the channel names like this: `biosignal.set_channel_name('Old name', 'New name')`.

> **Note**: 
> Since everything is an object in LTBio, it is important users respect the getters and setters boudaries, in order to keep the internal
> state of ``Biosignal`` objects stable. Although Python has ways to go around it, we recomend you to follow the "We are all consenting adults"
> convention.

### `in` checker

To check if `biosignal` has a channel named `xx`, use:
```
if 'xx' in biosignal:
    ...
```

To check if `biosignal` is defined at noon of January 3, use:
```
if '2023-01-03 12:00' in biosignal:
    ...
```


