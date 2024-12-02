---
layout: distill
title: "Diffusion Meets Flow Matching: Two Sides of the Same Coin"
# permalink: /main/
description: "Flow matching and diffusion models are two popular frameworks in generative modeling. Despite seeming similar, there is some confusion in the community about their exact connection. In this post, we aim to clear up this confusion and show that <i>diffusion models and Gaussian flow matching are the same</i>, although different model specifications can lead to different network outputs and sampling schedules. That's great news, it means you can use the two frameworks interchangeably."
date: 2025-12-02
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting

authors:
  - name: Ruiqi Gao
    url: "https://ruiqigao.github.io/"
    affiliations:
      name: Google DeepMind
  - name: Emiel Hoogeboom
    url: "https://ehoogeboom.github.io/"
  - name: Jonathan Heek
    url: "https://scholar.google.nl/citations?user=xxQzqVkAAAAJ&hl=nl"
  - name: Valentin De Bortoli
    url: "https://vdeborto.github.io/"
  - name: Kevin P. Murphy
    url: "https://scholar.google.com/citations?user=MxxZkEcAAAAJ&hl=en"    
  - name: Tim Salimans
    url: "https://scholar.google.nl/citations?user=w68-7AYAAAAJ&hl=en"    
# must be the exact same name as your blogpost
bibliography: 2025-04-28-distill-example.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Overview
  - name: Sampling
  - name: Training
  - name: Diving deeper into samplers
  - name: SDE and ODE perspective
  - name: Closing takeaways

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

{% include figure.html path="assets/img/2025-04-28-distill-example/twotrees.jpg" class="img-fluid" %}

Flow matching is gaining popularity recently, due to the simplicity of its formulation and the  “straightness” of its induced sampling trajectories. This raises the commonly asked question:

<p align="center"><i>"Which is better, diffusion or flow matching?"</i></p>

As we will see, diffusion models and flow matching are *equivalent* (for the common special case that the source distribution used with flow matching corresponds to a Gaussian). So there is not a single answer to this question. In particular, we will show how to convert one formalism to another. Why does this equivalence matter? This allows you to mix and match techniques developed for the two frameworks. For example, after training a flow matching model, you can use either a stochastic or deterministic sampling method (in contrast to the common misunderstanding that flow matching is always deterministic). 

We will focus on  the most commonly used flow matching formalism  <d-cite key="lipman2022flow"></d-cite>, which is closely related to <d-cite key="liu2022flow,albergo2023stochastic"></d-cite>. Our purpose is not to recommend one approach over another. Both frameworks are valuable, each rooted in distinct theoretical perspectives. It is even more encouraging that they lead to the same algorithm in practice. Our goal is to help practitioners feel confident using the two frameworks interchangeably, while understanding the true degrees of freedom one has when tuning the algorithm -- regardless of what it’s called.

<!-- Our purpose is not to recommend one approach over another. Instead our goal is to explain similarities and differences between the methods, and to explain the degrees of freedom one has when tuning each algorithm. -->



## Overview

We start with a quick overview of the two frameworks.


### Diffusion models

A diffusion process gradually destroys an observed datapoint $$ \bf{x} $$ (such as an image) over time $$t$$, by mixing the data with Gaussian noise. The noisy data at time $$t$$ is given by a forward process:
$$
\begin{equation}
{\bf z}_t = \alpha_t {\bf x} + \sigma_t {\boldsymbol \epsilon}, \;\mathrm{where} \; {\boldsymbol \epsilon} \sim \mathcal{N}(0, {\bf I}).
\label{eq:forward}
\end{equation}
$$
$$\alpha_t$$ and $$\sigma_t$$ define the **noise schedule**. A noise schedule is called variance-preserving if $$\alpha_t^2 + \sigma_t^2 = 1$$. The noise schedule is designed in a way such that $${\bf z}_0$$ is close to the clean data, and $${\bf z}_1$$ is close to a Gaussian noise.

To generate new samples, we can "reverse" the forward process: We initialize the sample $${\bf z}_1$$ from
a standard Gaussian. Given the sample $${\bf z}_t$$ at time step $$t$$, we predict what the clean sample might look like with a neural network  (a.k.a. denoiser model) $$\hat{\bf x} = \hat{\bf x}({\bf z}_t; t)$$, and then we project it back to a lower noise level $$s$$ with the same forward transformation:

$$
\begin{eqnarray}
{\bf z}_{s} &=& \alpha_{s} \hat{\bf x} + \sigma_{s} \hat{\boldsymbol \epsilon},\\
\end{eqnarray}
$$
where $$\hat{\boldsymbol \epsilon} = ({\bf z}_t - \alpha_t \hat{\bf x}) / \sigma_t$$.
(Alternatively we can train a neural network to predict the noise  $$\hat{\boldsymbol \epsilon}$$.)
We keep alternating between predicting the clean data, and projecting it back to a lower noise level until we get the clean sample.
This is the DDIM sampler <d-cite key="song2020denoising"></d-cite>. The randomness of samples only comes from the initial Gaussian sample, and the entire reverse process is deterministic. We will discuss the stochastic samplers later. 

### Flow matching

In Flow Matching, we view the forward process as a linear  interpolation between the data $${\bf x}$$
and a noise term $$\boldsymbol \epsilon$$:
$$
\begin{eqnarray}
{\bf z}_t = (1-t) {\bf x} + t {\boldsymbol \epsilon}.\\
\end{eqnarray}
$$

This corresponds to the diffusion forward process if the noise is Gaussian (a.k.a. Gaussian flow matching) and we use the schedule $$\alpha_t = 1-t, \sigma_t = t$$.

Using simple algebra, we can derive that  $${\bf z}_t = {\bf z}_{s} + {\bf u} (t - s) $$ for $$s < t$$, where $${\bf u} = {\boldsymbol \epsilon} - {\bf x}$$ is the "velocity", "flow", or "vector field". Hence, to sample
$${\bf z}_s$$ given $${\bf z}_t$$,  we reverse time and replace the vector field with our best guess at time $$t$$: 
$$\hat{\bf u} = \hat{\bf u}({\bf z}_t; t) = \hat{\boldsymbol \epsilon} - \hat{\bf x}$$,
represented  by a neural network, to get


$$
\begin{eqnarray}
{\bf z}_{s} = {\bf z}_t + \hat{\bf u}(s - t).\\
\label{eq:flow_update}
\end{eqnarray}
$$

Initializing the sample $${\bf z}_1$$ from a standard Gaussian, we keep getting $${\bf z}_s$$ at a lower noise level than $${\bf z}_t$$, until we obtain the clean sample.

### Comparison

So far, we can already discern the similar essences in the two frameworks:

<div style="padding: 10px 10px 10px 10px; border-left: 6px solid #FFD700; margin-bottom: 20px;">
  <p>1. <strong>Same forward process</strong>, if we assume that one end of flow matching is Gaussian, and the noise schedule of the diffusion model is in a particular form. </p>
  <p  style="margin: 0;">2. <strong>"Similar" sampling processes</strong>: both follow an iterative update that involves a guess of the clean data at the current time step. (Spoiler: below we will show they are exactly the same!)</p>
</div>


## Sampling 

It is commonly thought that the two frameworks differ in how they generate samples: Flow matching sampling is deterministic with "straight" paths, while diffusion model sampling is stochastic and with "curved paths". Below we clarify this misconception.
We will focus on deterministic sampling first which is simpler; we discuss the stochastic case later on.

 Imagine you want to use your trained denoiser model to transform random noise into a datapoint. Recall that the DDIM update is given by $${\bf z}_{s} = \alpha_{s} \hat{\bf x} + \sigma_{s} \hat{\boldsymbol \epsilon}$$. Interestingly, rearranging terms it can be expressed in the following formulation, with respect to several sets of network outputs and reparametrizations:

$$
\begin{equation}
\tilde{\bf z}_{s} = \tilde{\bf z}_{t} + \mathrm{Network \; output} \cdot (\eta_s - \eta_t) \\
\end{equation}
$$

| Network Output  | Reparametrization   |
| :------------- |-------------:|
| $$\hat{\bf x}$$-prediction      |    $$\tilde{\bf z}_t = {\bf z}_t / \sigma_t$$ and $$\eta_t = {\alpha_t}/{\sigma_t}$$ |
| $$\hat{\boldsymbol \epsilon}$$-prediction      |    $$\tilde{\bf z}_t = {\bf z}_t / \alpha_t$$ and $$\eta_t = {\sigma_t}/{\alpha_t}$$ |
| $$\hat{\bf u}$$-flow matching vector field      |    $$\tilde{\bf z}_t = {\bf z}_t/(\alpha_t + \sigma_t)$$ and $$\eta_t = {\sigma_t}/(\alpha_t + \sigma_t)$$ | 


Recall the flow matching update in Equation (4), look similar? If we set network output as $$\hat{\bf u}$$ in the last line and let $$\alpha_t = 1- t$$, $$\sigma_t = t$$, we have $$\tilde{\bf z}_t = {\bf z}_t$$ and $$\eta_t = t$$, so that we recover the flow matching update! More formally, the flow matching update can be considered the Euler integration of the reparametrized sampling ODE (i.e., $$\mathrm{d}\tilde{\bf z}_t = \mathrm{[Network \; output]}\cdot\mathrm{d}\eta_t$$), and


<div style="padding: 10px 10px 10px 10px; border-left: 6px solid #FFD700; margin-bottom: 20px;">
  <p align="center" style="margin: 0;"><em>Diffusion with DDIM sampler == Flow matching sampler (Euler).</em></p>
</div>


Some other comments on the DDIM sampler:


1. The DDIM sampler *analyically* integrates the reparametrized sampling ODE if the network output is a *constant* over time. Of course the network prediction is not constant, but it means the inaccuracy of DDIM sampler only comes from approximating the intractable integral of the network output (unlike the Euler sampler of the probability flow ODE <d-cite key="song2020score"></d-cite> which involves an additional linear term of $${\bf z}_t$$). The DDIM sampler can be considered an Euler integrator of the repamemetrized sampling ODE, which has the same update rule for different network outputs. However, if one uses a higher-order ODE solver, the network output can makes a difference, which means the $$\hat{\bf u}$$ output proposed by flow matching can make a difference from diffusion models.

2. The DDIM sampler is *invariant* to a linear scaling applied to the noise schedule $$\alpha_t$$ and $$\sigma_t$$,
as a scaling does not affect $$\tilde{\bf z}_t$$ and $$\eta_t$$. This is not true for other samplers e.g. Euler sampler of the probability flow ODE.

To validate Claim 2, we present the results obtained using several noise schedules, each of which follows a flow-matching schedule ($$\alpha_t = 1-t, \sigma_t = t$$) with different scaling factors. Feel free to change the slider. At the left end, the scaling factor is $$1$$ which is exactly the flow matching schedule, while at the right end, the scaling factor is $$1/[(1-t)^2 + t^2]$$, which corresponds to a variance-preserving schedule.
We see that DDIM (and flow matching sampler) always gives the same final data samples, regardless of the scaling of the schedule. The paths bend in different ways as we are showing $${\bf z}_t$$ (but not $$\tilde{\bf z}_t$$), which is scale-dependent along the path. For the Euler sampler of the probabilty flow ODE, the scaling makes a true difference: we see that both the paths and the final samples change.


<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-distill-example/interactive_alpha_sigma.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>


Wait a second? It is often said that the flow matching results in *straight* paths, but in the above figure its sampling trajectories look *curved*.

So why is flow matching said to result in straight sampling paths?
If the model would be perfectly confident about the data point it is moving to, the path from noise to data will be a straight line, with the flow matching noise schedule.
Straight line ODEs would be great because it means that there is no integration error whatsoever.
Unfortunately, the predictions are not for a single point. Instead they average over a larger distribution. And flowing *straight to a point != straight to a distribution*.


In the interactive graph below, you can change the variance of the data distribution on the right hand side by the slider.
Note how the variance preserving schedule is better (straighter paths) for wide distributions,
while the flow matching schedule works better for narrow distributions.

<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-distill-example/interactive_vp_vs_flow.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>


Finding such straight paths for real-life datasets like images is of course much less straightforward. But the conclusion remains the same: The optimal integration method depends on the data distribution.

Two important takeaways from deterministic sampling:

<div style="padding: 10px 10px 10px 10px; border-left: 6px solid #FFD700; margin-bottom: 20px;">
  <p>1. <strong>Equivalence in samplers</strong>: DDIM is equivalent to the flow matching sampler, and is invariant to a linear scaling to the noise schedule. </p>
  <p  style="margin: 0;">2. <strong>Straightness misnomer</strong>: Flow matching schedule is only straight for a model predicting a single point. For realistic distributions other interpolations can give straighter paths.</p>
</div>

<!-- 1. For DDIM the interpolation between data and noise is irrelevant and always equivalant to flow matching <d-footnote>The variance exploding formulation ($\alpha_t = 1$, $\sigma_t = t$) is also equivalant to DDIM and flow matching. -->

## Training 

<!-- For training, a neural network is estimated to predict $$\hat{\boldsymbol \epsilon} = \hat{\boldsymbol \epsilon}({\bf z}_t; t)$$ that effectively estimates $${\mathbb E} [{\boldsymbol \epsilon} \vert {\bf z}_t]$$, the expected noise added to the data given the noisy sample. Other **model outputs** have been proposed in the literature which are linear combinations of $$\hat{\boldsymbol \epsilon}$$ and $${\bf z}_t$$, and $$\hat{\boldsymbol \epsilon}$$ can be derived from the model output given $${\bf z}_t$$.  -->

Diffusion models <d-cite key="kingma2024understanding"></d-cite> are trained by estimating $$\hat{\bf x} = \hat{\bf x}({\bf z}_t; t)$$, or alternatively $$\hat{\boldsymbol \epsilon} = \hat{\boldsymbol \epsilon}({\bf z}_t; t)$$ with a neural net.
<!--In practice, one chooses a linear combination of $$\hat{\bf x}$$ and $${\bf z}_t$$ for stability reasons.-->
<!-- <d-footnote>It take a little bit of effort to show that indeed you only need linear combinations to define model outputs such as $$\hat{\boldsymbol{\epsilon}}$$, $$\hat{\bf v}$$ and $$\hat{\bf u}$$ (from flow matching)</d-footnote>. -->
Learning the model is done by minimizing a weighted mean squared error (MSE) loss:
$$
\begin{equation}
\mathcal{L}(\mathbf{x}) = \mathbb{E}_{t \sim \mathcal{U}(0,1), \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \left[ \textcolor{green}{w(\lambda_t)} \cdot \frac{\mathrm{d}\lambda}{\mathrm{d}t} \cdot \lVert\hat{\boldsymbol \epsilon} - {\boldsymbol \epsilon}\rVert_2^2 \right],
\end{equation}
$$
where $$\lambda_t = \log(\alpha_t^2 / \sigma_t^2)$$ is the log signal-to-noise ratio, and $$\textcolor{green}{w(\lambda_t)}$$ is the **weighting function**, balancing the importance of the loss at different noise levels. The term $$\mathrm{d}\lambda / {\mathrm{d}t}$$ in the training objective seems unnatural and in the literature is often merged with the weighting function. However, their separation helps *disentangle* the factors of training noise schedule and weighting function clearly, and helps emphasize the more important design choice: the weighting function.  

Flow matching also fits in the above training objective. Recall below is the conditional flow matching objective
used by <d-cite key="lipman2022flow, liu2022flow"></d-cite> :

$$
\begin{equation}
\mathcal{L}_{\mathrm{CFM}}(\mathbf{x}) = \mathbb{E}_{t \sim \mathcal{U}(0,1), \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \left[ \lVert \hat{\bf u} - {\bf u} \rVert_2^2 \right]
\end{equation}
$$

Since $$\hat{\bf u}$$ can be expressed as a linear combination of $$\hat{\boldsymbol \epsilon}$$ and $${\bf z}_t$$, the CFM training objective can be rewritten as mean squared error on $${\boldsymbol \epsilon}$$ with a specific weighting. 



### How do we choose what the network should output?

Below we summarize several network outputs proposed in the literature, including a few versions
used by diffusion models and the one used by flow matching. They can be derived from each other given the current data $${\bf z}_t$$. One may see the training objective defined with respect to MSE of different network outputs in the literature. From the perspective of training objective, they all correspond to having some additional weighting in front of the $${\boldsymbol \epsilon}$$-MSE that can be absorbed in the weighting function. 

| Network Output  | Formulation   | MSE on Network Output  |
| :------------- |:-------------:|-----:|
| $$\hat{\boldsymbol \epsilon}$$-prediction      |$$\hat{\boldsymbol \epsilon}$$ | $$\lVert\hat{\boldsymbol{\epsilon}} - \boldsymbol{\epsilon}\rVert_2^2 $$|
| $$\hat{\bf x}$$-prediction      | $$\hat{\bf x} = ({\bf x}_t - \sigma_t \hat{\boldsymbol \epsilon}) / \alpha_t $$      | $$ \lVert\hat{\bf x} - {\bf x}\rVert_2^2 = e^{-\lambda} \lVert\hat{\boldsymbol \epsilon} - {\boldsymbol \epsilon}\rVert_2^2 $$ |
| $$\hat{\bf v}$$-prediction | $$\hat{\bf v} = \alpha_t \hat{\boldsymbol{\epsilon}} - \sigma_t \hat{\bf x} $$      |    $$ \lVert\hat{\bf v} - {\bf v}\rVert_2^2 = \alpha_t^2(e^{-\lambda} + 1)^2 \lVert\hat{\boldsymbol \epsilon} - {\boldsymbol \epsilon}\rVert_2^2 $$ |
| $$\hat{\bf u}$$-flow matching vector field | $$\hat{\bf u} = \hat{\boldsymbol{\epsilon}} - \hat{\bf x} $$      |    $$ \lVert\hat{\bf u} - {\bf u}\rVert_2^2 = (e^{-\lambda / 2} + 1)^2 \lVert\hat{\boldsymbol \epsilon} - {\boldsymbol \epsilon}\rVert_2^2 $$ |

In practice, however, the model output might make a difference. For example,

* $$\hat{\boldsymbol \epsilon}$$-prediction can be problematic at high noise levels, because any error in $$\hat{\boldsymbol \epsilon}$$ will get amplified in $$\hat{\bf x} = ({\bf x}_t - \sigma_t \hat{\boldsymbol \epsilon}) / \alpha_t $$, as $$\alpha_t$$ is close to 0. It means that small changes create a large loss under some weightings. 

* Following the similar reason, $$\hat{\bf x}$$-prediction is problematic at low noise levels, because $$\hat{\bf x}$$ is not informative, and the error gets amplified in $$\hat{\boldsymbol \epsilon}$$.

Therefore, a heuristic is to choose a network output that is a combination of $$\hat{\bf x}$$- and $$\hat{\boldsymbol \epsilon}$$-predictions, which applies to the $$\hat{\bf v}$$-prediction and the flow matching vector field $$\hat{\bf u}$$.

### How do we choose the weighting function?

The weighting function is the most important part of the loss, it balances the importance of high frequency and low frequency components in perceptual data such as images, videos and audo <d-cite key="dieleman2024spectral,kingma2024understanding"></d-cite>. This is crucial, as certain high frequency components in those signals are not perceptible to humans, and thus it is better not to waste model capacity on them when the model capacity is limited. Viewing losses via their weighting, one can derive the following non-trivial result:

<div style="padding: 10px 10px 10px 10px; border-left: 6px solid #FFD700; margin-bottom: 20px;">
  <!-- <p>For weighting functions,</p> -->
  <p align="center" style="margin: 0;"><em>Flow matching weighting == diffusion weighting of ${\bf v}$-MSE loss + cosine noise schedule.</em></p>
</div>

That is, the conditional flow matching objective in Equation (7) is the same as a commonly used setting in diffusion models! See Appendix D.2-3 in <d-cite key="kingma2024understanding"></d-cite> for a detailed derivation. Below we plot several commonly used weighting functions in the literature, as a function of $$\lambda$$. 

<div class="m-page">
  <iframe src="{{ 'assets/html/2025-04-28-distill-example/weighting_functions.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>

The flow matching weighting (also $${\bf v}$$-MSE + cosine schdule weighting) decreases exponentially as $$\lambda$$ increases. Empirically we find another interesting connection: The stable diffusion 3 weighting <d-cite key="esser2024scaling"></d-cite>, a reweighted version of flow matching, is very similar to the EDM weighting <d-cite key="karras2022elucidating"></d-cite> that is popular for diffusion models.


### How do we choose the training noise schedule?

We discuss the training noise schedule in the last, as it should be the least important to training in the following sense:
1. The training loss is *invariant* to the training noise schedule. Specifically, the loss fuction can be rewritten as $$\mathcal{L}(\mathbf{x}) = \int_{\lambda_{\min}}^{\lambda_{\max}} w(\lambda) \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \left[ \|\hat{\boldsymbol{\epsilon}} - \boldsymbol{\epsilon}\|_2^2 \right] \, d\lambda$$, which is only related to the endpoints ($$\lambda_{\max}$$, $$\lambda_{\min}$$), but not the schedule $$\lambda_t$$ in between. In practice, one shoud choose $$\lambda_{\max}$$, $$\lambda_{\min}$$ such that the two ends are close enough to the clean data and Gaussian noise respectively. $$\lambda_t$$ might still affect the variance of the Monte Carlo estimator of the training loss. A few heuristics have been proposed in the literature to automatically adjust the noise schedules over the course of training. [This blog post](https://sander.ai/2024/06/14/noise-schedules.html#adaptive) has a nice summary.
2. Similar to sampling noise schedule, the training noise schedule is invariant to a linear scaling, as one can easily apply a linear scaling to $${\bf z}_t$$ and an unscaling at the network input to get the equivalence. The key defining property of a noise schedule is the log signal-to-noise ratio $$\lambda_t$$.
3. One can choose completely different noise schedules for training and sampling, based on distinct heuristics: For training, it is desirable to have a noise schedule that minimizes the variance of the Monte Carlo estimator, whereas for sampling the noise schedule is more related to the discretization error of the ODE / SDE sampling trajectories and the model curvature.

### Summary

A few takeaways for training of diffusion models / flow matching:

<div style="padding: 10px 10px 10px 10px; border-left: 6px solid #FFD700; margin-bottom: 20px;">
  <p>1. <strong>Equivalence in weightings</strong>: Weighting function is important for training, which balances the importance of different frequency components of perceptual data. Flow matching weighting is the same as a commonly used diffusion training weighting. </p>
  <p>2. <strong>Insignificance of training noise schedule</strong>: Noise schedule is far less important to the training objective, but can affect the training efficiency.</p>
  <p style="margin: 0;">3. <strong>Difference in network outputs</strong>: The network output proposed by flow matching is new, which nicely balances $\hat{\bf x}$- and $\hat{\epsilon}$-prediction, similar to $\hat{\bf v}$-prediction.</p>
</div>



## Diving deeper into samplers

In this section, we discuss different kinds of samplers in more detail.

### Reflow operator

The Reflow operation in Flow Matching connects noise and data points in a straight line.
One can obtain these (data, noise) pairs by running a deterministic sampler from noise.
A model can then be trained to directly predict the data given the noise avoiding the need for sampling.
In the diffusion literature, the same approach was the one of the first distillation techniques <d-cite key="luhman2021knowledge"></d-cite>.



### Deterministic sampler vs. stochastic sampler

So far we have just discussed the deterministic sampler of diffusion models or flow matching. An alternative is to use stochastic samplers such as the DDPM sampler <d-cite key="ho2020denoising"></d-cite>.

Performing one DDPM sampling step going from $\lambda_t$ to $\lambda_t + \Delta\lambda$ is exactly equivalent to performing one DDIM sampling step to $\lambda_t + 2\Delta\lambda$, and then renoising to $\lambda_t + \Delta\lambda$ by doing forward diffusion. That is, the renoising by doing forward diffusion reverses exactly half the progress made by DDIM. To see this, 
 let's take a look at a 2D example. 
Starting from the same mixture of Gaussians distribution, we can take either a small DDIM sampling step with the sign of the update reversed (left), or a small forward diffusion step (right):
{% include figure.html path="assets/img/2025-04-28-distill-example/particle_movement.gif" class="img-fluid" %}
For individual samples, these updates behave quite differently: the reversed DDIM update consistently pushes each sample away from the modes of the distribution, while the diffusion update is entirely random. However, when aggregating all samples, the resulting distributions after the updates are identical. Consequently, if we perform a DDIM sampling step (without reversing the sign) followed by a forward diffusion step, the overall distribution remains unchanged from the one prior to these updates. 
<!-- That means we can run DDIM update with a large step then followed by a "renoising" step, which matches the effect of running DDIM update with a smaller step.  -->


The fraction of the DDIM step to undo by renoising is a hyperparameter which we are free to choose (i.e. does not have to be exact half of the DDIM step), and which has been called the level of _churn_ by <d-cite key="karras2022elucidating"></d-cite>. Interestingly, the effect of adding churn to our sampler is to diminish the effect on our final sample of our model predictions made early during sampling, and to increase the weight on later predictions. This is shown in the figure below:


<div class="l-page" style="display: flex; justify-content: center; align-items: center; height: 100%;">
  <iframe src="{{ 'assets/html/2025-04-28-distill-example/churn.html' | relative_url }}" 
          frameborder="0" 
          scrolling="no" 
          height="600px" 
          width="80%">
  </iframe>
</div>


Here we ran different samplers for 100 sampling steps using a cosine noise schedule
and $$\hat{\bf v}$$-prediction <d-cite key="salimansprogressive"></d-cite>. Ignoring nonlinear interactions, the final sample produced by the sampler can be written as a weighted sum of predictions $$\hat{\bf v}_t$$ made during sampling and a Gaussian noise $${\bf e}$$: $${\bf z}_0 = \sum_t h_t \hat{\bf v}_t +  \sum_t c_t {\bf e} $$. The weights $$h_t$$ of these predictions are shown on the y-axis for different diffusion times $$t$$ shown on the x-axis. DDIM results in an equal weighting of $$\hat{\bf v}$$-predictions for this setting, as shown in <d-cite key="salimansprogressive"></d-cite>,
whereas DDPM puts more emphasis on predictions made towards the end of sampling. Also see <d-cite key="lu2022dpm"></d-cite> for analytic expressions of these weights in the $$\hat{\bf x}$$- and $$\hat{\boldsymbol \epsilon}$$-predictions.



## SDE and ODE Perspective

<!-- So far, we have shown that the equivalence of the flow matching sampler and the DDIM sampler. 
We have also shown that the weightings appearing in flow matching and diffusion models can all be expressed in a general framework by expressing them in terms of log-SNR. 
This should (hopefully!) have convinced you that the frameworks are identical. 
But how easily can you move from one framework to the other?  -->

We've observed the practical equivalence between diffusion models and flow matching algorithms. Here, we formally describe the equivalence of the forward process and sampling using ODE and SDE, as a completeness in theory. 
<!-- derive <strong>exact formula</strong> to move from a diffusion model to a flow matching perspective and vice-versa.  -->

### Diffusion Model

<!-- We have stated in the overview that "A diffusion process gradually destroys an observed data $$ \bf{x} $$ over time $$t$$". But what is this gradual process? 

It can be fully described by the following evolution equation -->

The forward process of diffusion models which gradually destroys a data over time can be described by the following stochastic differential equation (SDE):

$$
\begin{equation}
\mathrm{d} {\bf z}_t = f_t {\bf z}_t \mathrm{d} t + g_t \mathrm{d} {\bf z} ,
\end{equation}
$$

where $$\mathrm{d} {\bf z}$$ is an <em> infinitesimal Gaussian</em> (formally, a Brownian motion).
$f_t$ and $g_t$ decide the noise schedule. The generative process is given by the reverse of the forward process, whose formula is given by 

$$
\begin{equation}
\mathrm{d} {\bf z}_t = \left( f_t {\bf z}_t - \frac{1+ \eta_t^2}{2}g_t^2 \nabla \log p_t({\bf z_t}) \right) \mathrm{d} t + \eta_t g_t \mathrm{d} {\bf z} ,
\end{equation}
$$

where $\nabla \log p_t$ is the <em>score</em> of the forward process. 
<!-- (This is why you might have noticed that some papers refer to diffusion models as "score-based generative models".) -->

Note that we have introduced an additional parameter $\eta_t$ which controls the amount of stochasticity at inference time. This is related to the <em>churn</em> parameter introduced before. When discretizing the backward process we recover DDIM in the case $\eta_t = 0$ and DDPM in the case $\eta_t = 1$. 

<!-- So to summarize there are three free parameters:

1. $f_t$ which controls how much we forget the original data in the forward process.
2. $g_t$ which controls how much noise we input into the samples in the forward process.
3. $\eta_t$ which controls the amount of stochasticity at inference time. -->

### Flow Matching

The interpolation between $${\bf x}$$ and $${\boldsymbol \epsilon}$$ in flow matching can be described by the following ordinary differential equation (ODE):

$$
\begin{equation}
\mathrm{d}{\bf z}_t = {\bf u}_t \mathrm{d}t.
\end{equation}
$$

Assuming the interpolation is $${\bf z}_t = \alpha_t {\bf x} + \sigma_t {\boldsymbol \epsilon}$$, then $${\bf u}_t = \dot{\alpha}_t {\bf x} + \dot{\sigma}_t {\boldsymbol \epsilon}$$.

The generative process is simply reversing the ODE in time, and replacing $${\bf u}_t$$ by its conditional expectation with respect to $${\bf z}_t$$. This is a specific case of <em>stochastic interpolation</em><d-cite key="liu2022flow,albergo2023stochastic"></d-cite>, in which case it can be generalized to an SDE:

$$
\begin{equation}
\mathrm{d} {\bf z}_t = ({\bf u}_t - \frac{1}{2} \varepsilon_t^2 \nabla \log p_t({\bf z_t})) \mathrm{d} t + \varepsilon_t \mathrm{d} {\bf z},
\end{equation}
$$
where $$\varepsilon_t$$ controls the amount of stochasticity at inference time.

<!-- To summarize,   flow matching frameworks can be entirely determined by three hyperparameters  

<div style="padding: 10px 10px 10px 10px; border-left: 6px solid #FFD700; margin-bottom: 20px;">
  <p>1. $\alpha_t$ which controls the data component in the interpolation. </p>
  <p>2. $\sigma_t$ which controls the noise component in the interpolation. </p>
  <p style="margin: 0;">3. $\varepsilon_t$ which controls the amount of stochasticity at inference time. </p>
</div> -->

### Equivalence of the two frameworks

<!-- Despite their clear similarities it is not immediately clear how to link the diffusion model framework and the flow matching one. 
Below, we provide formulae which give a one-to-one mapping between the two frameworks. In short:

<div style="padding: 10px 10px 10px 10px; border-left: 6px solid #FFD700; margin-bottom: 20px;">
  Diffusion model and flow matching are just one change of variable away!
</div> -->
Both frameworks are defined by three hyperparameters respectively: $f_t, g_t, \eta_t$ for diffusion, and $\alpha_t, \sigma_t, \varepsilon_t$ for flow matching. We can show the equivalence by deriving one set of hyperparameters from the other. From diffusion to flow matching:

$$
\alpha_t = \exp\left(\int_0^t f_s \mathrm{d}s\right) , \quad \sigma_t = \left(\int_0^t g_s^2 \exp\left(-2\int_0^s f_u \mathrm{d}u\right) \mathrm{d} s\right)^{1/2} , \quad \varepsilon_t = \eta_t g_t . 
$$


<!-- Similarly, given a flow matching framework, i.e. hyperparameters $\alpha_t, \sigma_t, \varepsilon_t$
one can derive the equivalent difusion model by defining -->


From flow matching to diffusion:

$$
f_t = \partial_t \log(\alpha_t) , \quad g_t^2 = 2 \alpha_t \sigma_t \partial_t (\sigma_t / \alpha_t) , \quad \eta_t = \varepsilon_t / (2 \alpha_t \sigma_t \partial_t (\sigma_t / \alpha_t))^{1/2} . 
$$

In summary, aside from training considerations and sampler selection, diffusion and Gaussian flow matching exhibit no fundamental differences.

## Closing takeaways

If you've read this far, hopefully we've convinced you that diffusion models and Gaussian flow matching are equivalent. However, we highlight two new model specifications that flow matching brings to the field:

* **Network output**: Flow matching proposes a vector field parametrization of the network output that is different from the ones used in diffusion literature. The network output can make a difference when  higher-order sampler is used. It may also affect the training dynamics. 
* **Sampling noise schedule**: Flow matching leverages a simple sampling noise schedule $$\alpha_t = 1-t$$ and $$\sigma_t = t$$, with the same update rule as DDIM.

It would be interesting to investigate the importance of these two model specifications empirically in different real world applications, which we leave to the future work.

## Acknowledgements

Thanks to our colleagues at Google DeepMind for fruitful discussions. In particular, thanks to Sander Dieleman and Ben Poole. 



