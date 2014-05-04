Hybrid analysis for spike sorting
=================================
NOTICE:
We provide a dataset with ground truth against which any spike sorting algorithm can be tested (e.g. masked KlustaKwik) can be downloaded from here:

[NCHybrid136](https://googledrive.com/host/0BwTrbfNJNihcWWpKeHpaYUJmWjg)

The files are in the kwik format.

There are 138572 points (spikes) of 96 dimensions. 

The ground truth dataset was constructed from two datasets kindly supplied by Mariano Belluscio and Gyorgy Buzsaki,
made with a zigzag 32-channel probe in the rat cortex.

The mean waveform from a single recording was digitally added to the second recording at specified times with the amplitude of the hybrid spike was varied randomly between 50% and 100% of its original value. Here we have only added the final `hybrid dataset' called NCHybrid136.kwd. The spike waveforms can be found in the file NCHybrid136.kwx and the groundtruth can be found in the file NCHybrid136.kwik.


* [Overview of the system](http://nbviewer.ipython.org/urls/raw.github.com/klusta-team/hybrid_analysis/master/notes/Overview.ipynb)
* [Detailed structure](http://nbviewer.ipython.org/urls/raw.github.com/klusta-team/hybrid_analysis/master/notes/Detailed_structure.ipynb)

* NEVER produce the Overview.tex file manually, always use:

        ipython nbconvert --to latex Overview.ipynb --post PDF
