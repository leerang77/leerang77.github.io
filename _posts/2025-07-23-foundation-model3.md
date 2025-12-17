---
layout: post
title: Fine-tuning protein language model with Huggingface (Part 3)
date: 2025-12-12 06:00:00
description: Code package and training example
tags: protein-language-model
categories: Bio-ML
featured: false
related_posts: false
---

# **Intro**
In [Part 2](https://leerang77.github.io/blog/2025/foundation-model2/) of this series, I broke down the key components of the pipeline for fine-tuning protein language models. 

In this final part, we will share the final code package and examples of training results.

# **Code Package**

The full code package is available my on my github: [PLM_finetune](https://github.com/leerang77/PLM_finetune). While I have added some additional functionalities and structure, the core components remain the same as what I have described in the previous posts. The installation and usage instructions are available in detail in the `README.md` file.

To very briefly summarize the usage:

## Installation
Clone the repository and install the package using pip:
```bash
pip install -e .
```
Make sure to also install the required dependencies listed in `requirements.txt` using your package manager of choice. Also make sure that a compatible CUDA version is installed if you plan to use GPU acceleration.

## Data Preparation
The jupyter notebook `data/prepare_data.ipynb` downloaded and preprocesses the datasets for each of the four residue/protein + classiffication/regression task combinations.

## Config files
Next, define the config file for the experiment. Example configs are available in the `plft/configs/` folder as .yaml files. You can modify them as needed. 

## Running the experiment
Finally, run the training using the defined config file, e.g.
```bash
python -m plft.pipeline.py --config-name protbert_chezod_token_regression.yaml trainer.learning_rate=1e-5 trainer.batch_size=16
```

# **Training Results**
I will highlight two example tasks here: protein-level classification of subcellular location (SCL), and residue-level regression of protein disorder. These example tasks were taken from Schmirler et al. 2024, which took it from other sources. The original sources of the datasets are cited at the end. 

## Dataset

### 1. Protein Subcellular Location (SCL) Classification
The SCL dataset is downloaded from [FLIP Benchmark Datset](https://github.com/J-SNACKKB/FLIP/tree/main/splits/scl), and are originally from [DeepLoc-1](https://doi.org/10.1093/bioinformatics/btx431) and [DeepLoc-2](https://doi.org/10.1093/nar/gkac278). It contains protein sequences labeled with one of 10 subcellular locations: Cytoplasm, Nucleus, Cell membrane, Mitochondrion, Endoplasmic reticulum, Lysosome/Vacuole, Golgi apparatus, Peroxisome, Extracellular and Plastid.
<div class="row justify-content-center mt-3">
  <div class="col-md-6">
    {% include figure.liquid loading="eager" path="assets/img/2025-07-23-foundation-model/SCL.jpg" class="img-fluid" %}
    <div class="caption mt-2 text-center">
      Figure 1: Distributioon of protein subcellular locations in the SCL dataset.
    </div>
  </div>
</div>

We used the ProtBert model as the backbone pLM, and fine-tuned it using LoRA for the SCL classification task. We compared its performance against a frozen backbone ProtBert with an MLP head. The results are shown in Figure 1. We can see that the LoRA fine-tuned ProtBert outperforms the frozen backbone ProtBert with MLP head across the eval loss, accuracy, and F1 metrics. 

<div class="row justify-content-center mt-3">
  <div class="col-md-6">
    {% include figure.liquid loading="eager" path="assets/img/2025-07-23-foundation-model/SCL_plots.jpg" class="img-fluid" %}
    <div class="caption mt-2 text-center">
      Figure 2: Comparing the performance of LoRA fine-tuned ProtBert against frozen backbone ProtBert with MLP head, on the SCL classification task.
    </div>
  </div>
</div>


### 2. Residue Disorder Regression
The SETH dataset quantifies residue disorder from NMR chemical shifts. This dataset originates from the CheZOD1174 subset of the [SETH project](https://doi.org/10.3389/fbinf.2022.1019597). For each residue, SETH computes a Z-score of the observed NMR chemical shift deviation from typical structured values.

* Low order score ( < 8) → intrinsically disordered residue or flexible region 
* High order score (> 8) → ordered residue (stable secondary structure)


<div class="row justify-content-center mt-3">
  <div class="col-md-6">
    {% include figure.liquid loading="eager" path="assets/img/2025-07-23-foundation-model/Chezod.jpg" class="img-fluid" %}
    <div class="caption mt-2 text-center">
      Figure 3. Distribution of residue disorder scores in the SETH dataset.
    </div>
  </div>
</div>

We again used the ProtBert model as the backbone pLM, and fine-tuned it using LoRA for the residue disorder regression task. The results are shown in Figure 4. We can see that again, the LoRA fine-tuned ProtBert outperforms the frozen backbone ProtBert with MLP head for the mean absolute error. 

<div class="row justify-content-center mt-3">
  <div class="col-md-6">
    {% include figure.liquid loading="eager" path="assets/img/2025-07-23-foundation-model/Chezod_plots.jpg" class="img-fluid" %}
    <div class="caption mt-2 text-center">
      Figure 4: Comparing the performance of LoRA fine-tuned ProtBert against frozen backbone ProtBert with MLP head, on the residue disorder regression task.
    </div>
  </div>
</div>

# **Citations**
