# Video Generation via Deep GRU U-Net with DDPM Correction

This is an attempt to perform Deep Video Prediction via an Autoregressive ConvGRU and DDPM architecture. This is based off of (https://arxiv.org/pdf/2203.09481.pdf).

## Method

Within the paper, the primary sampling and prediction is performed by an Autoregressive model, which aims to model $$ p(x_{0:T}) = \Pi^T_{t=0} (x_t|x_{<t})$$
This is performed through the hidden state model: $$ p(x_{0:T}) = \Pi^T_{t=0} p(x_t|h_t)p(h_t|h_{<t}, x_{<t})$$
This however produces inadequate results, thus the paper opts to use a DDPM to perform residual correction, allowing the old model to model a premliminary prediction mu:
$$ p(\mu_t |x_{<t}) $$
Then the DDPM will predict a corrective term (y), in which the following relationship aims to be held for some hyperparameter sigma:
$$ y_{\theta}^t =\frac{x^t-\mu_\phi(x_{<t})}{\sigma}$$
From there during sampling, the corrections can be applied, by prediction the true x, off of the incorrect previous prediction, and the DDPM residual correction:
$$ \sigma y_{\theta}^t+\mu_\phi(x_{<t}) =x^t$$

## Current Results

Currently there are architectural issues with the Autoregressive model. Currently the trained version of the model lacks the ability to accurately model predictions on the MovingMNIST dataset which I chose to test on:


![Prediction](https://i.imgur.com/gJI4IV6.png)
![Ground Truth](https://i.imgur.com/iM7gaHp.png)

This is likely due to the architecture not having adequate size, or the missing residual connections within the implementation within the upsample/downsample blocks. This should be corrected, before moving on to the DDPM residuals, as the DDPM should mainly perform small corrections.
