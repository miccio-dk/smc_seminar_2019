# Introduction
Virtual reality (VR) and augmented reality (AR) research has made substantial progress over the last decades, and virtual environments created using binaural sound rendering technologies find applications in a wide array of areas, such as aids for the visually impaired, tools for audio professionals, and VR-based entertainment systems.

These techniques are based on the application of a particular filter called head-related transfer functions (HRTFs), which colors a sound according to its location in the virtual environment.
However, HRTFs derived from standard anthropometric pinnae, such as those in dummy heads, often results in localization errors and wrong spatial perception [@moller_binaural_1996].
In fact, while generic HRTFs may successfully approximate the interaural time difference (ITD) and interaural level difference (ILD) cues which are used to perceive the horizontal direction of a sound source, the monaural cues needed to discern its vertical direction are highly dependent on the anthropometric characteristics of each ear.

In order to provide the most realistic and immersive experience possible, it is necessary for users to have their custom set of HRTFs measured, which can prove quite impractical due to the need for dedicated facilities and the overall invasiveness of the procedure.
Recently, attempts have been performed at synthesizing or customizing HRTFs using various data from users such as anthropometric measurements, 3D scans, or perceptual feedback.


# State of the art
Over the past decades, several strategies have been devised, in order to avoid the burden of conducting strenuous acoustical measurements with human subjects.
In a recent review, Guezenoc [@guezenoc_hrtf_2018] divides such alternative approaches into _numerical simulation_, _anthropometrics_-based, and _perceptual feedback_-based.

The former method consists in simulating the propagation of acoustic waves around the subject, using 3D scans; the most common simulation schemes include Fast-Multipole-accelerated Boundary Element Method (FM-BEM) [@gumerov_fast_2007] and Finite Difference Time Domain (FDTD) [@mokhtari_comparison_2007] for frequency and time domain respectively.

With the help of databases of publicly available HRTFs and machine learning techniques, anthropometric measurements can be used to choose, adapt, or estimate a subject's HRTF set.
In 2010, Zeng [@zeng_hybrid_2010] implements an hybrid model based on principal component analysis (PCA) and multiple linear regression, which uses anthropometric parameters to select the most suitable HRTF set for the given user.
Similarly, user feedback on perceptual tests can be used to inform regression models for tasks such as those listed above.

In more recent times, there has been an interest in solving the aforementioned tasks using deep learning techniques.
In 2017, Yao [@yao_head-related_2017] uses anthropometric measurements to select the most suitable HRTF sets from a larger database.
In 2018, Lee [@lee_personalized_2018] develops a double-branched neural network that processes anthropometric data with a multi-layer perceptron and edge-detected pictures of the ear with convolutional layers, combining the outputs of the two into a third network to estimate HRTF sets.
Again in 2017, Yamamoto [@yamamoto_fully_2017] trains a variational autoencoder on HRTF data, and devises a perceptual calibration procedure to fine-tune the latent variable used as input by the generative part of the model.
Finally, in 2019, Chen et al. [@chen_autoencoding_2019] train an autoencoder to reconstruct HRTFs along the horizontal plane, and subsequently uses the resulting latent representations as targets for a multilayer perceptron which feeds on anthropometric data and azimuth angle, allowing users to synthesize new HRTFs using the MLP and decoder.



# Milestone plan
With regards to the work of Yamamoto [@yamamoto_fully_2017], I will set to:

- Fully understand the methods employed
- Implement VAE with relevant novel features
- Replicate training of VAE with HRTF data
- Inspect resulting latent space for coherent mapping of features
- Replicate parameter-tuning mechanism, or devise novel one (e.g. based on anthropometric measurements)
- Validate on secondary dataset (HUTUBS, VIKING HRTF, etc
- Evaluate alternative input representations, such as Real Spherical Harmonic [@romigh_efficient_2015]


# Bibliography
