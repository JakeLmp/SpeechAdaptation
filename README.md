# Repository for Speech Adaptation MVPA pipelines

Time-Generalized MVPA (decoding), like in
-  King, J.-R., & Dehaene, S. (2014). Characterizing the dynamics of mental representations: The temporal generalization method. Trends in Cognitive Sciences, 18(4), 203–210. https://doi.org/10.1016/j.tics.2014.01.002
- Heikel, E., Sassenhagen, J., & Fiebach, C. J. (2018). Time-generalized multivariate analysis of EEG responses reveals a cascading architecture of semantic mismatch processing. Brain and Language, 184, 43–53. https://doi.org/10.1016/j.bandl.2018.06.007

Written with [MNE-Python](https://mne.tools/stable/index.html "https://mne.tools/stable/index.html") and [scikit-learn](https://scikit-learn.org/stable/index.html "https://scikit-learn.org/stable/index.html"). Mainly intended for use with EEG data, could relatively easily be adapted to other data.

## MVPA data format requirements

See MNE import documentation for BrainVision files
- MNE importing in general: https://mne.tools/stable/auto_tutorials/io/20_reading_eeg_data.html#brainvision-vhdr-vmrk-eeg
- BrainVision only: https://mne.tools/stable/generated/mne.io.read_raw_brainvision.html#mne.io.read_raw_brainvision
- ```.vhdr``` files are detected, accompanying files (```.vmrk```, ```.eeg```) should be named identically for the import to work.

Files provided to the module should contain:
- Preprocessed data channels
  - Artifacts removed, filtered, etc.
  - Artifact-containing trials may be included in the file, as long as they are marked as such. The module will automatically exclude these trials.
  - Preprocessing can also be done with MNE, but is not implemented here.
- *Exclude* EOG and reference channels
  - Module accounts for this, but life's easier when only the channels of interest are included.
- *One* participant per file
- *All* experiment conditions in one file, with corresponding stimulus markers
- Optionally downsampled (e.g. 100 Hz) to compensate for high computational load of MVPA
  - Module also contains an option to do this.

## How to run

First, open the package's ```PARAMETERS.toml``` file, edit the parameter values to your liking, and *save the file*. Then, run the following command in the terminal:

```unix
python -m MVPA
```

Inspect the ERP plots to see if data preprocessing was performed correctly, and wait for the analysis to complete. Results are stored in (subdirectories of) the user-specified directory.