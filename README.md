Hybrid analysis for spike sorting
=================================
NOTICE:
For the a dataset with ground truth against which any spike sorting algorithm can be tested (e.g. masked KlustaKwik) can be downloaded from here:

[NCHybrid136](https://googledrive.com/host/0BwTrbfNJNihcWWpKeHpaYUJmWjg)

The files are in the kwik format.

* [Overview of the system](http://nbviewer.ipython.org/urls/raw.github.com/klusta-team/hybrid_analysis/master/notes/Overview.ipynb)
* [Detailed structure](http://nbviewer.ipython.org/urls/raw.github.com/klusta-team/hybrid_analysis/master/notes/Detailed_structure.ipynb)

* NEVER produce the Overview.tex file manually, always use:

        ipython nbconvert --to latex Overview.ipynb --post PDF
