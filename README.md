# Learning Probability Density Functions Using GAN

## Overview

This project focuses on learning an unknown probability density function
(PDF) from data alone using a Generative Adversarial Network (GAN). The
dataset used contains NO₂ concentration levels collected across India.
Instead of assuming a known distribution, the GAN implicitly learns the distribution directly from
transformed samples.

------------------------------------------------------------------------
## Dataset

**Source:** India Air Quality Data (Kaggle)

**Feature Used:** NO₂ concentration

### Preprocessing Steps

1.  Converted the NO₂ column to numeric values.
2.  Removed non-numeric entries using coercion.
3.  Dropped missing values.
4.  Filtered out negative readings.
5.  Converted the cleaned data into NumPy format for further processing.

------------------------------------------------------------------------

## Transformation Function

Each NO₂ value `x` is transformed into `z` using:

z = x + a_r \* sin(b_r \* x)

Where:

-   a_r = 0.5 \* (r mod 7)
-   b_r = 0.3 \* ((r mod 5) + 1)

**Roll Number: 102317229 (here)**

**Computed Parameters:**
-  a_r = 0.0
-  b_r = 1.5

------------------------------------------------------------------------

## Methodology

### Why GAN?

Traditional density estimation techniques require assuming a parametric
distribution. GANs eliminate this assumption and instead learn the data
distribution implicitly through adversarial training.

------------------------------------------------------------------------

## GAN Architecture

### Generator

A fully connected neural network designed to map Gaussian noise into
realistic samples of the transformed variable.

**Structure:**

Input (Noise \~ N(0,1)) → Linear(1,64) → ReLU → Linear(64,64) → ReLU →
Linear(64,1)

**Purpose:** Produce synthetic samples that resemble the real
transformed data.

------------------------------------------------------------------------

### Discriminator

A binary classifier that distinguishes between real transformed samples
and generator-produced samples.

**Structure:**

Input → Linear(1,64) → LeakyReLU → Linear(64,64) → LeakyReLU →
Linear(64,1) → Sigmoid

**Purpose:** Improve generator performance by providing adversarial
feedback.

------------------------------------------------------------------------

## Training Configuration
### Hyperparameter   Value

  Epochs:           5000\
  Batch Size:       128\
  Optimizer:        Adam\
  Learning Rate:    0.0002\
  Loss Function:    Binary Cross Entropy

### Training Procedure

1.  Sample real transformed data.
2.  Generate fake samples using Gaussian noise.
3.  Train discriminator on real vs fake data.
4.  Train generator to fool the discriminator.
5.  Repeat for multiple epochs until equilibrium is reached.

------------------------------------------------------------------------

## Result Graph

The final probability density was estimated using:

-   Histogram Density
-   Kernel Density Estimation (KDE)

The KDE curve provides a smooth approximation of the learned
distribution and aligns closely with the histogram, indicating that the
generator successfully modeled the data distribution.

------------------------------------------------------------------------

## Result Analysis

### Mode Coverage

The generator captures the major modes of the transformed distribution
without significant mode collapse. This suggests effective adversarial
learning.

### Training Stability

The discriminator and generator losses move toward equilibrium during
training, indicating stable convergence where neither network dominates
the other.

### Quality of Generated Distribution

The generated samples closely resemble the structure of the real
transformed data. Since no parametric assumptions were made, this
demonstrates the GAN's ability to perform implicit density estimation.

------------------------------------------------------------------------
## Conclusion

This project demonstrates that Generative Adversarial Networks can
effectively learn an unknown probability density function using only
sample data. The generator successfully models the transformed NO₂
distribution, and KDE confirms the quality of the learned density.
