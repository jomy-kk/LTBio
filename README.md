## IT - Long Term Biosignals Framework

[![Test Biosignals Package](https://img.shields.io/pypi/v/LongTermBiosignals)](https://pypi.org/project/LongTermBiosignals)

[![Test Biosignals Package](https://github.com/jomy-kk/IT-PreEpiSeizures/actions/workflows/test-biosignals-package.yml/badge.svg?branch=main&event=push)](https://github.com/jomy-kk/IT-PreEpiSeizures/actions/workflows/test-biosignals-package.yml)
 [![Test Processing Package](https://github.com/jomy-kk/IT-PreEpiSeizures/actions/workflows/test-processing-package.yml/badge.svg)](https://github.com/jomy-kk/IT-PreEpiSeizures/actions/workflows/test-processing-package.yml)
[![Features](https://github.com/jomy-kk/IT-PreEpiSeizures/actions/workflows/test-features-package.yml/badge.svg)](https://github.com/jomy-kk/IT-PreEpiSeizures/actions/workflows/test-features-package.yml)
[![Machine Learning](https://github.com/jomy-kk/IT-PreEpiSeizures/actions/workflows/test-ml-package.yml/badge.svg)](https://github.com/jomy-kk/IT-PreEpiSeizures/actions/workflows/test-ml-package.yml)
[![Decision](https://github.com/jomy-kk/IT-PreEpiSeizures/actions/workflows/test-decision-package.yml/badge.svg)](https://github.com/jomy-kk/IT-PreEpiSeizures/actions/workflows/test-decision-package.yml)
[![Pipeline](https://github.com/jomy-kk/IT-LongTermBiosignals/actions/workflows/test-pipeline-package.yml/badge.svg)](https://github.com/jomy-kk/IT-LongTermBiosignals/actions/workflows/test-pipeline-package.yml)
[![Integration Tests](https://github.com/jomy-kk/IT-LongTermBiosignals/actions/workflows/test-integration.yml/badge.svg)](https://github.com/jomy-kk/IT-LongTermBiosignals/actions/workflows/test-integration.yml)

#### Description
Python library for easy managing and processing of large Long-Term Biosignals.
This repository is object of evaluation of some Master's and Doctoral's theses.

#### Contribute!

* 🪲 Report bugs <a href="https://github.com/jomy-kk/IT-LongTermBiosignals/issues/new?assignees=jomy-kk&labels=fix&template=bug_report.md&title=%5BBUG%5D+Write+a+title+here">here</a>.
* 💡 Suggest features <a href="https://github.com/jomy-kk/IT-LongTermBiosignals/issues/new?assignees=jomy-kk&labels=enhancement&template=feature_request.md&title=%5BNEW%5D+Write+a+title+here">here</a>.
* 💬 Community Q&A <a href="https://github.com/jomy-kk/IT-LongTermBiosignals/discussions/categories/q-a">here</a>.

#### Informal Documentation

📑 Acess to the <a href="https://github.com/jomy-kk/IT-LongTermBiosignals/wiki">Wiki</a>.

Full reference documentarion comming soon.

## Getting Started

#### Installing the Package

This framework was developed and tested on Python 3.10.4, so make sure you have an interperter >= 3.10.4.

If you are familired with pip, you can download and install the package by running:

```
pip install LongTermBiosignals
```

If not, you may download the latest stable GitHub release [here](https://github.com/jomy-kk/IT-LongTermBiosignals/releases) and place a copy of the `ltbio` directory (which is inside `src`) on your project's root.

#### Dependencies

See Python dependencies in `requirements.txt`.

You may consider installing the following on your machine if needed:
* `graphviz` to plot Pipeline diagrams (`sudo apt-get install graphviz` or `brew install graphviz`)
* `h5py` to read HDF5 files if running on an Apple Sillicon machine (`brew install hdf5 && export HDF5_DIR=/opt/homebrew/bin/brew/Cellar/hdf5/<version>`)

#### Simple Use Case

Let's create a sequence of samples using `Timeseries`:

```
from ltbio.biosignals import Timeseries

ts = Timeseries([1, 2, 3, 4, 5], initial_datetime = datetime.datetime.now(), sampling_frequency = 360.0)
```

Add let's pretend this was a 1-lead ECG 🫀:

```
from ltbio.biosignals.modalities import ECG
 
my_ecg = ECG({'Left': ts})
```

Simple, right? There's loads of more stuff to customize like sources, body locations, physical units, events, etc. Explore it in this [notebook](https://github.com/jomy-kk/IT-LongTermBiosignals/blob/main/examples/getting_started.ipynb) 📓 and in the <a href="https://github.com/jomy-kk/IT-LongTermBiosignals/wiki">Wiki</a>.

________

#### Copyright Notice

There is no license for the content of this repository.

2022, All rights reserved. This means content of this repository cannot be reproduced nor distributed, and no work can be derived from this work without explicit and written permission from the authors.
