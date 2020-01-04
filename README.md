# HRTF Individualization using Deep Learning
> Material from _Research in Sound and Music Computing (SMC9) CPH_

This repository contains the outcomes of a seminar course in Sound and Music Computing, and comprises a seminar paper, presentation slides, and a software implementation.


## Structure

### Seminar paper
The directory `seminar_paper` contains the source code for the seminar paper.
The paper has been authored in Markdown, and converted in PDF using Pandoc.
Its textual content is enclosed in `paper.md`, while the list of references can be found in `bibliography.bib`.
A copy of the document can be compiled by running `make` from within the directory; the output can be found in `build/paper.pdf`.

### Seminar presentation
The directory `seminar_presentation` contains the source code for the seminar slides.
The paper has been authored in Markdown, and converted into HTML+reveal.js slides using Pandoc.
Its textual content is enclosed in `slides.md`, while the interactive slides can be navigated by opening `slides.html` in a browser.
A copy of the document can be compiled by running `make` from within the directory.

### Seminar codebase
The directory `seminar_codebase` contains the software implementation for the replication of literature research, as well as further experiments.
It comprises the following:

- `utils_data.py`: a collection of functions for loading and processing data (HRTFs, ear pictures, anthropometric measurements, etc)
- `utils_model.py utils_model_1d.py utils_train.py`: a collection of functions for generating and training deep learning models 
- `utils_plot.py`: a collection of functions for showing the results (latent space visualization, comparison of reconstructed HRTFs or ear pictures, correlation matrices, etc)
- a series of Jupiter notebooks, constituting the experiments, where all the aforementioned scripts are used:
  - `vae_hutubs.ipynb`: autoencoding HRTFs (2d elevation-azimuth representation)
  - `vae_hutubs_hrtf.ipynb`: autoencoding HRTFs (2d frequency-elevation representation)
  - `vae_hutubs_gpu1.ipynb`: preliminary experiments with autoencoding pinna depth maps
  - `vae_hutubs_ears.ipynb`: autoencoding pinna depth maps using Inception layers
  - `vae_hutubs_chen2019.ipynb`: autoencoding individual HRTFs using dense or 1D-convolutional layers
  - `vae_hutubs_3d.ipynb`: autoencoding HRTFs patches and reconstructing one-hot representation
  - `pca_dnn.ipynb`: predicting HRTF principal components using anthropometric measurements (k-fold validation)
  - `ear_pca_dnn.ipynb`: predicting HRTF principal components using anthropometric measurements and principal components from pinna depth maps (k-fold validation)


## Instructions

### Setup
- Install `conda` (an environment and dependency manager for Python)
- Create environment: `conda env create -f environment.yml`
- Run Jupyter: `jupyter-lab`

### Usage
Notebooks can be accessed from the browser at `127.0.0.1:8888`.
The cells within the notebooks contains precomputed output related to the latest execution.
However, each notebook has hyperparameters that can be adjusted.
Unfortunately, most notebooks are not entirely polished, presenting unused code, duplicated sections, and various data, reflecting the exploratory and experimental nature of the research.

When attempting to run any of the code, access to the datasets is necessary.
These can be copied (e.g. using `scp`) from the remote location `/home/rmicci18/hrtf_i/data/` in the ML workstation and onto `seminar_codebase/data/`.
More information on how to access the ML workstation remotely and  how to copy data from it can be found [here](https://app.gitbook.com/@aalborg-university/s/machine-learning-workstation).


**Riccardo Miccini, 2019-2020**

