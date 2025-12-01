# 👕 Variational Autoencoder (VAE) for Fashion-MNIST

A deep learning project implementing a Convolutional Variational Autoencoder (VAE) using TensorFlow/Keras. This VAE is trained on the **Fashion-MNIST dataset** to learn a compressed, continuous **latent space** for generating and reconstructing clothing images.

---

## 🚀 Project Overview

This project aims to demonstrate the capabilities of VAEs in learning robust feature representations and generating new data samples. We experimented with various configurations, focusing on the trade-off between **reconstruction fidelity** and **latent space regularization**.

### Key Features:
* **Convolutional Architecture:** Utilizes Conv2D layers for efficient feature extraction.
* **Latent Dimensionality:** Explored models with $D_h=2$, $D_h=32$, and the optimal $D_h=32$ configuration.
* **Loss Function Comparison:** Directly compares the performance of **Mean Squared Error (MSE)** and **Binary Crossentropy (BCE)** losses.
* **KLD Warm-Up:** Implements a warm-up schedule for the KL Divergence term to ensure stable training and prevent posterior collapse.

---

## 🧠 Model Architecture (High-Capacity $D_h=32$)

The final robust VAE architecture employs deep convolutional layers to manage the complexity of the Fashion-MNIST dataset.

### 1. Encoder
The Encoder maps the input image $(28 \times 28 \times 1)$ down to the latent parameters $\mu$ and $\log(\sigma^2)$.

* Input: $28 \times 28 \times 1$
* Layers: `Conv(32, s=2)` $\rightarrow$ `Conv(64, s=2)` $\rightarrow$ `Conv(128, s=1)` $\rightarrow$ `Flatten` $\rightarrow$ `Dense(256)`
* Output: $\mathbf{z}_{\mu}$ and $\mathbf{z}_{\log \sigma^2}$ (both $1 \times 32$)

### 2. Decoder
The Decoder reconstructs the image from a sampled latent vector $\mathbf{z}$. Residual blocks were introduced to enhance image quality and detail preservation.

* Input: $\mathbf{z}$ ($1 \times 32$)
* Layers: `Dense(7x7x128)` $\rightarrow$ `Reshape` $\rightarrow$ `Conv2DTranspose` (Upsampling) $\rightarrow$ **Residual Blocks**
* Output: $28 \times 28 \times 1$ (Activation: `sigmoid`)

---

## 📈 Key Results and Experiments

### 1. MSE with High Latent Dimension ($D_h=32$)
The initial objective was met using MSE. While the model achieved a continuous and well-structured latent space, the reconstructions suffered from the characteristic **"VAE blurriness"** inherent to the Gaussian noise assumption of the MSE loss function.

### 2. ⭐ Bonus: Sharp VAE (BCE + Low $\beta$)
To overcome the blurriness, a dedicated experiment was conducted using **Binary Crossentropy (BCE)** as the reconstruction loss, alongside a low KL weight ($\beta=0.01$).

| Feature | MSE Model | Sharp VAE (BCE) |
| :--- | :--- | :--- |
| **Loss Function** | MSE (Mean/Summed) | **BCE (Summed)** |
| **KL Weight ($\beta$)** | Optimized (e.g., 0.5-1.0) | **Low (0.01)** |
| **Reconstruction** | Blurry/Washed-Out | **High Contrast & Sharpness** |
| **Latent Space** | Highly Regular | Slightly less regular (prioritizes fidelity) |

**Conclusion:** The BCE model provided superior **perceptual quality**, demonstrating a direct trade-off between the mathematical regularity (KL term) and the visual fidelity (Reconstruction term).

---
