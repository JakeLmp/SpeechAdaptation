# Repository for EEG MVPA pipelines

Time-Generalized MVPA (decoding), like in
-  King, J.-R., & Dehaene, S. (2014). Characterizing the dynamics of mental representations: The temporal generalization method. Trends in Cognitive Sciences, 18(4), 203–210. https://doi.org/10.1016/j.tics.2014.01.002
- Heikel, E., Sassenhagen, J., & Fiebach, C. J. (2018). Time-generalized multivariate analysis of EEG responses reveals a cascading architecture of semantic mismatch processing. Brain and Language, 184, 43–53. https://doi.org/10.1016/j.bandl.2018.06.007

And other approaches, like in

- Grootswagers, T., Wardle, S. G., & Carlson, T. A. (2017). Decoding Dynamic Brain Patterns from Evoked Responses: A Tutorial on Multivariate Pattern Analysis Applied to Time Series Neuroimaging Data. Journal of Cognitive Neuroscience, 29(4), 677–697. https://doi.org/10.1162/jocn_a_01068

Written with [MNE-Python](https://mne.tools/stable/index.html "https://mne.tools/stable/index.html") and [scikit-learn](https://scikit-learn.org/stable/index.html "https://scikit-learn.org/stable/index.html"). Mainly intended for use with EEG data, could relatively easily be adapted to other data.

## MVPA data format requirements

See MNE import documentation for BrainVision files
- MNE data imports in general: https://mne.tools/stable/auto_tutorials/io/20_reading_eeg_data.html#brainvision-vhdr-vmrk-eeg
- MNE BrainVision data imports specifically: https://mne.tools/stable/generated/mne.io.read_raw_brainvision.html#mne.io.read_raw_brainvision
- TL;DR: ```.vhdr``` files are detected, accompanying files (```.vmrk```, ```.eeg```) should be named identically for the import to work.

Files provided to the tool should contain:
- *One* participant per file
- *All* experiment conditions in one file, with corresponding stimulus markers
- Preprocessed data channels
  - Artifacts removed, filtered, etc.
  - Artifact-containing trials may be included in the file, as long as they are marked as such. The package will automatically exclude these trials.
  - Preprocessing can also be done with MNE, but is not implemented here.
- *Exclude* EOG and reference channels
  - package accounts for this, but life's easier when only the channels of interest are included.
- Optionally downsampled (e.g. 100 Hz) to compensate for high computational load of MVPA
  - package also contains an option to do this.

## Installation

The repository can be downloaded using

```unix
git clone git@github.com:JakeLmp/SpeechAdaptation.git
```

and following the subsequent steps. Required third-party Python packages can be installed using

```unix
pip install -r requirements.txt
```



## How to run

1. Open the package's ```PARAMETERS.toml``` file, edit the parameter values to your liking, and ***save the file***. 
2. Set the repository as the current working directory.
3. Run the following command in the terminal:

```unix
python -m MVPA
```

4. Inspect the ERP plots (located in the ```results/plot``` directory) to see if data preprocessing was performed according to expectations, and wait for the tool to complete. 

Results are stored in (subdirectories of) the user-specified directory.

For more options running the tool, type

```unix
python -m MVPA --help
```

to print usage instructions.

## Plotting

Plots of the results are automatically generated and stored in the ```results/plot``` directory. You may want to explore the data further using other plotting parameters. Examples on how to use the included plotting functions are included in the ```plotting_expamples.ipynb``` notebook. 