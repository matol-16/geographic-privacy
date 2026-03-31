# Protecting images from Diffusion based geolocalization

Project for the class *Multimodal Generative AI* taught by Professor Vicky Kalogetion at Ecole Polytechnique (2026).

## Repository Organisation

This repository is a fork from the repository of the paper [Around the World in 80 Timesteps: A Generative Approach to Global Visual Geolocation](https://github.com/nicolas-dufour/plonk) (Dufour et al., 2025).

It is aimed at researching adversarial methods to prevent images from being localizable by the diffusion and flow matching models presented by Dufour et al. (2025).

All the new contributions are in the the *adversarial_demo* folder, which is added at the root of the *plonk* project.

Occasional additions are made to the requirements.txt file.

In order to install librairies and get the code running, refer to the README_ORIGINAL_PAPER.md file. Below, we propose a simple method to download dependencies, and present the code's structure

## Installation

From the original repository's readme, we suggets the following installation routine:

'''
conda create -n plonk python=3.10
conda activate plonk
pip install -e .
'''

## Code structure

As mentioned above, all the new code in is *adversarial_demo*. It is organized as follows:

**Evaluation scripts**:
- *eval_notebook_attacks.ipynb* is a simple notebook that allows testing our adversarial framework on an image saved in plonk/.media
- *scripts_eval.py* allows running several evaluations (attack final step displacement, link between attack success and localizability...). Evaluations can be very long (several tens of hours on an A5000 GPU whithout parallelization) because one needs to train every attack for multiple attack budgets on multiple images.



**Code used**:
- *adversarial_eval.py* implements the code to evaluate our attacks on YFCC4K and OSV-5M.
- *adversarial_metrics.py* and *adversarial_utils.py* provide basic functions sued throughout our code. The metrics implemented are the ones described in section 4 of our report
- *attacks.py* coordinates the attack scripts, which are contained in *encoder_attacks.py* and *trajectory_attacks.py* for Encoder and Diffusion Trajectory Deviation respectively. It provides a single method to evaluate both attacks on an image.
- *pipe_trajectory* implements a hereditary class of *Plonk*'s *PlonkPipeline* class, which returns diffusion trajectories when calling the pipeline. This allowed testing our attack's effect on whole trajectories, not just predicted locations.
- *plots_adversarial_attacks.py* provides the functions to plot the figures that present the results of attack evaluationsin the paper.
- *build_yfcc4k_from_revisiting_im2gps.py* allows downloading the YFCC4K dataset (only needs to run once)

**Others**
- Evaluation results are stored in the *results* folder. One can run directly our plotting methods that retrieve stored results.
- *archive_code* consists of code used during the exploratory phase.
