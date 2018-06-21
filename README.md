Hybrid analysis for spike sorting
=================================
NOTICE:
We provide a dataset with ground truth against which any spike sorting algorithm can be tested (e.g. masked KlustaKwik) can be downloaded from here:

[NCHybrid136](https://drive.google.com/open?id=0BwTrbfNJNihcWWpKeHpaYUJmWjg)

The files are in the kwik format.

There are 138572 points (spikes) of 96 dimensions. 

The ground truth dataset was constructed from two datasets kindly supplied by Mariano Belluscio and Gyorgy Buzsaki,
made with a zigzag 32-channel probe in the rat cortex.

The mean waveform from a single recording was digitally added to the second recording at specified times with the amplitude of the hybrid spike varied randomly between 50% and 100% of its original value to model a busrting neuron. Here we have only supplied the final `hybrid dataset' called NCHybrid136.kwd. The spike waveforms can be found in the file NCHybrid136.kwx and the groundtruth clustering can be found in the file NCHybrid136.kwik. In the groundtruth clustering, 
a 1 indicates an added hybrid spike whereas a 0 indicates a background spike. The efficacy of any spike sorting algorithm can be measured by whether it is able to assign all the the hybrid spikes into a single cluster and a single cluster alone. 

Using the analysis code
=================================

* [Overview of the system](http://nbviewer.ipython.org/urls/raw.github.com/klusta-team/hybrid_analysis/master/notes/Overview.ipynb)
* [Detailed structure](http://nbviewer.ipython.org/urls/raw.github.com/klusta-team/hybrid_analysis/master/notes/Detailed_structure.ipynb)

* NEVER produce the Overview.tex file manually, always use:

        ipython nbconvert --to latex Overview.ipynb --post PDF
