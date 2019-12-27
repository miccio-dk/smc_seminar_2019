# Introduction {#intro}
[//]: # (motivations)
Virtual reality (VR) and augmented reality (AR) research has made substantial progress over the last decades, and virtual environments created using binaural sound rendering technologies find applications in a wide array of areas, such as aids for the visually impaired, tools for audio professionals, and VR-based entertainment systems.

[//]: # (key concepts)
These techniques are based on the application of a particular filter called _head-related transfer function_ (HRTF), which colors a sound according to its location in the virtual environment.
However, HRTFs derived from standard anthropometric pinnae, such as those in dummy heads, often results in localization errors and wrong spatial perception [@moller_binaural_1996].
In fact, while generic HRTFs may successfully approximate the interaural time difference (ITD) and interaural level difference (ILD) cues which are used to perceive the horizontal direction of a sound source, the monaural cues needed to discern its vertical direction are highly dependent on the anthropometric characteristics of each ear.

In order to provide the most realistic and immersive experience possible, it is necessary for users to have their custom set of HRTFs measured, which can prove quite impractical due to the need for dedicated facilities and the overall invasiveness of the procedure.
Recently, attempts have been performed at synthesizing or customizing HRTFs using various data from users such as anthropometric measurements, 3D scans, or perceptual feedback.

[//]: # (research question, roadmap)
This paper investigates methods for generating individualized HRTFs, in particular using newly-developed deep learning algorithms, and further expands on the topic by documenting the replication attempts and experiments conducted as part of the research.
In the next section, current relevant contributions to the field are introduced.
The [Methods](#methods) section details the computational techniques used in selected works from the literature as well as in the research carried out as part of the seminar, with particular focus on deep learning methods.
In [Results](#results), the applications and outcomes of the aforementioned techniques for replications and other experiments are discussed, with the purpose of assessing their effectiveness. 
Finally, closing remarks as well as pointers for future research are stated in the [Conclusions](#end) section.

# State of the art {#sota}
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

# Methods {#methods}
This section presents some of the most relevant computational methods found in the relevant literature on HRTF individualization.
The aspects covered in the following subsections include the encoding of generated HRTFs, the extraction and choice of predictors, and the deep neural network architectures adopted.

## HRTF representation {#repr}
single HRTF, 2d repr, 3d repr, patch [@yamamoto_fully_2017], principal components

## Anthropometric data extraction {#anthr}
FABIAN dataset [fabian_hutubs_2019], depthmaps, CIPIC [algazi_cipic_2001] anthropometric parameters

## Autoencoder {#ae}
Most conventional neural networks are employed to predict a target $y$ from an input $x$ in a supervised manner.
On the other hand, autoencoders learn a compressed representation $z$ of the input data $x$ called _latent representation_, which is then used to generate a reconstructed version $\hat{x}$.
The purpose of autoencoders is to learn useful features from the input data in an unsupervised manner [@goodfellow_deep_2016].

Autoencoders usually consists of a feed-forward neural network composed of two sub-nets: an encoder network $f()$ and a decoder network $g()$ such that $g(f(x)) = g(z) = \hat{x}$.
Training an autoencoder usually involves iteratively updating the weights and biases of the two networks through backpropagation, in order to minimize a cost function representing the mean squared error (MSE) between $x$ and $\hat{x}$.

Over time, several variants of the autoencoder have been developed.
Each variant extends on the conventional autoencoder architecture by promoting different properties of the latent space, thereby catering to different tasks such as denoising, classification, or --- as it is this case here --- generative applications. 
Two common generative models based on the autoencoder are described below.

### Variational autoencoder (VAE) {#vae}
Variational autoencoders are a class of generative models, extending the classic autoencoder.
A VAE is a probabilistic model where the encoder $q_{\theta}(z|x)$ maps the probability distribution a certain latent representation given a data point, and the decoder $p_{\phi}(x|z)$ outputs the probability distribution of the data, given a point in the latent space.
In VAEs, it is often desireable to model the latent space as an isotropic multivariate Gaussian distribution.
This constraint is enforced by introducing a measure of distance between the aforementioned prior distribution $p_{\theta}(z) \sim \mathcal{N} (0,1)$ and the encoder distribution, called Kullback-Leibler divergence.
This probabilistic framework proves useful when synthesizing HRTFs, because it can learn causal factors of variations in the data [@kingma_introduction_2019].
However, there exists no way of generating a specific HRTF --- i.e. one for a given combination of azimuth and elevation angles; while the points in latent space are likely to generate plausible new data, one can only obtain random instances of said data.
The class of autoencoders described below aims at addressing this shortcoming.

### Conditional variational autoencoder (CVAE) {#cvae}

## Architectures and models {#models}


# Results {#results}
Introduce the three main themes.


## Autoencoding ear images {#vae-ear}

## Autoencoding HRTFs {#vae-hrtf}

## Autoencoding principal components {#vae-pca}

### Predicting principal components {#dnn}



# Conclusions {#end}

 [@romigh_efficient_2015]


# Bibliography {#biblio}
