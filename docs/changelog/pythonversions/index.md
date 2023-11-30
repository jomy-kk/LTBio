# LTBio Python Library


## [1.1.0](404)

_Released: 05-02-2023 | Created: 20-01-2023 | **Public**_

> Spatial efficiency for Biosignals (only loaded from .biosignal files), by using memory maps.

### Added

### Changed

* When saving a `Biosignal` to a `.biosignal` file, the samples of all `Segment`'s are memory mapped to disk, and the `Biosignal` object is
dumped as that. The main advantage is that samples are stored in disk, and only coppied to memory when needed. This functionality is kept
when the `Biosignal` is loaded in future Python sessions.

### Deprecated

### Removed

### Fixed

____

## 1.0.2

_Released: 20-12-2022 | Created: 05-07-2022 | **Not Public**_

> Major bug fixes and more complex tests to ensure the library's stability.

### Added

* Class `Normalizer` as a formatter.

### Changed

### Deprecated

### Removed

### Fixed

____

## [1.0.1](https://github.com/jomy-kk/IT-LongTermBiosignals/releases/tag/v1.0.1)

_Released: 23-06-2022 | Created: 01-06-2022 | **Public**_

> First ready-to-use realease.

### Added

* New biosignal modalities: `ACC`, `ECG`, `EDA`, `EEG`, `EMG`, `PPG`, `RESP`, `TEMP`.
* New biosignal sources: `Bitalino`, `E4`, `HEM`, `HSM`, `MITDB`, `Seer`, `Sense` (as examples).
* New medical conditions: `Epilepsy` and `COVID19` (as examples).
* New surgical procedure: `CarpalTunnelRelease` (as example).
* New body locations: general anatomical location, and ECG and EEG electrode locations (as examples).
* New features: mean, variance, HRV, ... (as examples).
* Classes `OverlappingTimeseries` and `Frequency`.
* Class `SupervisedTrainReport` to produce PDF reports of ML models.
* Package `pipeline`: classes `Pipeline`, `PipelineUnit`, `SinglePipelineUnit`, `PipelineUnitsUnion` (`ApplyTogether` and `ApplySeparatly`), `Packet`, `Input`, `GoTo`.

### Changed
* Class `Filter` divided in two: `FrequencyDomainFilter` and `TimeDomainFilter`.

### Fixed

* Public API calls. Each submodule imports what should be used by the user.


_____

## 1.0.0

_Released: 31-05-2022 | Created: 01-02-2022 | **Not Public**_

> Begining of the LTBio Python Library. All packages were available, except ``pipeline``.

### Added

* Abstract classes `Biosignal` and `__BiosignalSource`, and some concrete implementations in the sub-packages
`modalities` and `sources`, respectively.
* Classes `Timeseries`, `Segment`, `Unit`, `Event`.
* Packages `clinical`: classes `Patient`, `BodyLocation`, `MedicalCondition`, `Medication`, and `SurgicalProcedure`.
* Package `processing`: classes `Segmenter` and `Filter`.
* Package `features`: classes `FeatureExtractor` and `FeatureSelector`.
* Package `ml`: classes `SupervisedModel`, `SupervisedTrainConditions`, `SurpervisingTrainer`, `SupervisedTrainResults`.
