# Documentation

## Feasibility Review

### User Need, Scope & Acceptance Criteria
The intended user are downstream model developers who require a generative model to produce syntetic data for training and testing their models while preserving patient pricacy.

The prototype will be a WGAN-GP model that can generate synthetic medical hip MRI images that are balanced in anatomical fidelity, distributional diversity, and privacy preservation.

The acceptance criteria for the prototype are:
1. SSIM > 0.6 against reference anatomical slices
2. Euclidean distance to training set > 5% of distance tonearest neighbors in training set
3. peak VRAM usage < 16GB and training time < 8h on A100 GPU

### Model Choice & Course Concepts
The model choice is a Wasserstein GAN with Gradient Penalty (WGAN-GP) which is a generative adversarial network that uses the Wasserstein distance as a loss function and includes a gradient penalty to enforce the Lipschitz constraint. This model is chosen because it has been shown to produce high-quality images and is more stable during training compared to traditional GANs.

The model will be trained on a dataset of 2D prostate MRI slices, with 11460 training, 540 test, and 660 validation images.

The original WGAN paper shows improved stability and convergence compared to traditional GANs, and the gradient penalty helps to enforce the Lipschitz constraint, which is important to prevent the discriminator gradient from exploding or vanishing. The WGAN-GP model has been shown to produce high-quality images and is more stable during training compared to traditional GANs or DCGANs.

### Preliminary Feasibility Evidence - Data Audit
The dataset has been audited for statistical properties, and the following summary statistics have been computed, indicating that the dataset is suitable for training a generative model. Mean and variance indicate a well distributed dataset, while skewness and kurtosis indicate that the dataset has a reasonable distribution of pixel intensities.
```
Mean                  : -0.6572
Variance              : 0.1244
Standard Deviation    : 0.3527
Skewness              : 1.4659
Excess Kurtosis       : 2.2119
```
The UMAP projection (`plots/real_mri_umap_audit.png`) of 50 PCA components also shows that the dataset is well distributed and does not have any obvious clusters or outliers. The top 10 outlier images (`plots/mri_outliers_audit.png`) have been identified and visually inspected, and they do not appear to be significantly different from the rest of the dataset.

### Risks, Budget & Fallback
Key risks:
1. Mode collapse or blurry anatomics -> enough epochs, model architecture research, data augmentation, audit SSIM
2. Patient scan memorization (overfitting) -> audit nearest neighbor distance and latent distance, use data augmentation and regularization

Training should be done as early as possibly given the probably long training period on rangpur. A maximum GPU time of 10 hours is estimated.

A workable fallback will be the baseline DCGAN which is simpler to implement.