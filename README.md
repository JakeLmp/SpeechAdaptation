# Repository for Speech Adaptation MVPA pipelines

Time-Generalized MVPA (decoding), like in
-  King, J.-R., & Dehaene, S. (2014). Characterizing the dynamics of mental representations: The temporal generalization method. Trends in Cognitive Sciences, 18(4), 203–210. https://doi.org/10.1016/j.tics.2014.01.002
- Heikel, E., Sassenhagen, J., & Fiebach, C. J. (2018). Time-generalized multivariate analysis of EEG responses reveals a cascading architecture of semantic mismatch processing. Brain and Language, 184, 43–53. https://doi.org/10.1016/j.bandl.2018.06.007

Written with MNE-Python and scikit-learn. Mainly intended for use with EEG data, could relatively easily be adapted to other data.

## MVPA data format requirements

See MNE import documentation
- general: https://mne.tools/stable/auto_tutorials/io/20_reading_eeg_data.html#brainvision-vhdr-vmrk-eeg
- but especially: https://mne.tools/stable/generated/mne.io.read_raw_brainvision.html#mne.io.read_raw_brainvision
- ```.vhdr``` files are detected, accompanying files (```.vmrk```, ```.eeg```) should be named identically --- except for the extension ---  for the import to work.

Data contained in each file:
- Preprocessed data channels
  - Artifact removal, filtering, etc.
  - Artifact-containing trials may be included in the import, as long as they are marked as such. The script will automatically exclude these trials.
  - This can also be done with MNE, but is not implemented here.
- *Exclude* EOG and reference channels
  - Script accounts for this, but life's easier when only the data of interest is included.
- *One* participant per file
- *All* conditions in one file, with corresponding stimulus markers
- Optionally downsampled (e.g. 100 Hz) to compensate for high computational load of MVPA
  - Script also contains an option to do this.