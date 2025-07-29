---
layout: post
title: Fine-tuning protein language model with Huggingface (Part 1)
date: 2025-07-23 15:09:00
description: Motivation for fine-tuning and intro to Huggingface
tags: protein-language-model
categories: Bio-ML
featured: false
related_posts: false
---

# **What is this post about?**
This post is a polished version of my learning about fine-tuning models on Huggingface, which I wrote some months ago. Therefore, it is likely most useful for someone with similar experience: I was working as computational biologist working with and developing ML models, had good understanding of protein language models but had primarily used them for generating embeddings rather than training them. I was not familiar with using Huggingface libraries to tune existing models and wanted to learn about it. But hopefully it will be useful for people with different backgrounds too.

## What it covers
1. What it means to fine-tune a protein language model and why you might want to do it (Part 1)
2. What Huggingface is and how it simplifies working with pretrained transformer models (Part 1)
3. Practical examples and workflow of fine-tuning code with Huggingface libraries (Part 2)
4. Parameter-efficient fine-tuning with Hugginface PEFT library (Part 2)

# **Motivation: Why fine tune a protein language model?**
Foundation models are becoming increasingly popular in biology across various domains. Models like [ESM series](https://www.evolutionaryscale.ai/blog/esm3-release) and [Geneformer](https://www.nature.com/articles/s41586-023-06139-9) have been trained on large-scale protein sequences and single-cell transcriptomics. Google Deepmind recnetly released [Tx-LLM](https://research.google/blog/tx-llm-supporting-therapeutic-development-with-large-language-models/), a large-language model based on PaLM2 that can predict properties across modalities (small molecules, proteins, nucleic acids, cell lines, diseases). In March, they followed up with an open-source release of [TxGemma](https://arxiv.org/abs/2504.06196), a smaller scale version of Tx-LLM. 

Protein language models (pLMs) like ESM pretrained on large sequence databases learn to map protein sequence to a vector embedding in a high-dimensional representation space. Thus, it becomes quantifiable how two protein sequences are similar along certain directions in the high-dimensional space while different in other dimensions. The embeddings capture the proteins’ structure, biophysical properties, and evolutionary context; different characteristics maps onto low-dimensional subspaces within the embedding. 

Therefore, these embeddings are very useful as input features for downstream property-prediction models. Suppose a supervised property-prediction model is trained on only hundreds of labeled sequences, which may seem too small for the model to generalize over the immense space of protein sequence and structure. If, however, the target property depends mainly on a few specific embedding dimensions and these labeled data adequately covers the distribution of natural proteins along those dimensions, the model could then accurately predict the property on novel sequences by examining the embedding values along those key dimensions. 

**If pLMs already learn to generate information-rich embeddings from pretraining on billions of sequences, why fine-tune the pLMs to a specific task?** How could this help? Couldn't it lead to overfitting and deterioriation of generizability? I recently came across a simple and elegant [paper](https://www.nature.com/articles/s41467-024-51844-2) by Burkhard Rost Lab that tested whether fine-tuning of pLMs coupled with head prediction models help improve the performance, using three foundation models (ESM2, ProtT5, Ankh) on eight different prediction tasks. The tasks included predicting protein mutational landscapes, stability, intrinsically disordered region, sub-cellular location, and secondary structure. Nearly all of the model-task combinations showed improvement with fine-tuning of the foundation model weights!

<div class="row justify-content-center mt-3">
  <div class="col-md-6">
    {% include figure.liquid loading="eager" path="assets/img/2025-07-23-foundation-model/Rost_1.jpg" class="img-fluid" %}
    <div class="caption mt-2 text-center">
      Figure 1: Reproduced from Schmirler et al. Shows the percentage differences between the fine-tuned and pretrained models for the eight prediction tasks (x-axis). Blue tiles mark statistically significant increases (>1.96 standard errors; fine-tuning better). See more details at Schmirler et al. 
    </div>
  </div>
</div>

**What's the physical intuition for why fine-tuning helps?** When we backpropagate through both the prediction head and the foundation model during supervised training with labeled data, two things happen at once, First, the prediction head learns which embedding dimensions or their mapping correspond to the target property. Second, the encoder is refined to encode sequences in a way that sequences are most cleanly projected onto the dimensions relevant to the target property. It’s not always obvious how much extra gain we get from updating the foundation model itself versus just training a head on frozen embeddings, but in principle this joint adjustment can yield a representation that’s better aligned to the prediction task.

## Fine-tuning vs. Domain-adaptive pretraining

I have seen the term "fine-tuning" used in at least two separate contexts in literature, so as a first step, I decided to look into it more:

- **Scenario 1:** A prediction task uses a foundation model embedding, and updates during supervised training backpropagates through both models. For example, in Schmirler et al. for predicting the mutational landscape of GFP protein, the mutant sequences are fed into ESM2 and those embeddings are fed into MLP classifier head. During the supervised training of the MLP head, the loss on mutational effect prediction is used to update the weights of both the MLP head and the ESM2 model.
- **Scenario 2:** This scenario takes a foundation model trained on wide corpus of sequences, and continues to train with the same objective but with a more specific dataset. For example [Madani et al](https://www.nature.com/articles/s41587-022-01618-2), first trains ProGen model is on 281 million protein sequences from UniParc, UniProtKB, Pfam, and NCBI. ProGen is a generative model and is trained on next-token prediction problem. Then, before it's used for generating artificial lysozyme sequences, it's fine-tuned with 55,948-sequences that belong to phage lysozyme (PF00959), pesticin (PF16754), glucosaminidase (PF01832), glycoside hydrolase family 108 (PF05838) and transglycosylase (PF06737) from the Pfam family. Notably, in this case the objective function for fine-tuning is still the next-token prediction loss.

These scenarios show that the term fine-tuning can be used broadly in the field. Here's are more precise definitions for distinguishing them:

- **Task-adaptive fine-tuning (TaFT)**
    - **What it is:** Pretrained foundation model is attached to a downstream prediction task model, and the weights are updated using a supervised loss for the downstream property-prediction task (i.e. Scenario 1)
    - **When to use it:** When there are enough labeled examples of the target property that the foundation representations themselves should shift to better separate the examples.
- **Domain-adaptive pretraining (DaPT)**
    - **What it is:** Pretrained foundation model continues to be trained on its original self-supervised objective (e.g. masked language modeling) but using a small, specialized dataset (i.e. Scenario 2)
    - **When to use it:** When the target proteins are from a niche family whose statistics differ substantially from the base model’s training set. The idea is to realign the foundation’s language to the domain of interest before any supervised step.

The key distinction is that domain-adaptive pre-training simply consists of training the model on its original objective. In contrast, task-adaptive fine-tuning requires connecting a foundation model with a prediction task head and backpropagating through both. **For the rest of this post, we will focus on task-adaptive fine-tuning.**

## Parameter-efficient fine-tuning

Before diving into the implementation of fine-tuning, let’s consider its two main downsides: cost and risk.

- **Cost:** Full fine-tuning of a foundation model with hundreds of millions to tens of billions of parameters can quickly become expensive and complicated. Training very large models will require multiple GPUs with distributed training and complex checkpointing workflows.
- **Risk:** Updating pLM weights based on limited supervised tasks opens the door to overfitting. pLM pretraining results in embeddings that contain rich structural and physicochemical information. After the fine-tuning updates, the model may end up forgetting these more fundamental information and instead memorizing the idiosyncrasies of the small training set. This phenomenon of losing the previously learned knowledge after fine-tuning is called catastrophic forgetting.

A strategy that can mitigate both of these problems is parameter efficient fine-tuning (PEFT). Common PEFT approaches include:

- **Partial fine-tuning**
    - Weights from only certain parts (e.g. the final transformer block) are updated, while the others are frozen
- **Adapters**
    - Additional layers are inserted within the model (rather than on top like the prediction head) and only these are trained, while the original model weights are frozen
- **Reparameterization (Low Rank Adaptation; LoRA)**
    - Updates to weight matrices are allowed only as a product of two lower rank matrices, drastically reducing the number of trainable parameters

There are many other resources to learn more about the methods of PEFT. [This blog post from IBM is a nice introduction.](https://www.ibm.com/think/topics/parameter-efficient-fine-tuning) In this post, I’ll mainly look at the implementation of LoRA.


# **Implementation of task-adaptive fine-tuning with Huggingface**

## Challenges of using other people’s models

As noted before, task-adaptive fine-tuning requires connecting a pretrained foundation model with a prediction task head and backpropagating through both. Suppose we try to implement it from scratch. Because it’s hard to predict which foundation model may generate the best embedding for our downstream task (see the figure from Schmirler et al: performance of different pLMs vary for downstream tasks), we want to test several pLMs. We will face the following time-consuming and tedious tasks to get started with the pLMs:

- Installing multiple pLM packages or cloning the repos
- Creating and managing environments for each model
- Reading through documentations of varying qualities and figuring out model-specific quirks, such as handling special tokens or understanding the correct arguments for the `forward()` method
- Reading through the codes to understand the quirks if documentation isn’t great

Adding prediction head and fine-tuning will create more headaches with: 

- Having to subclass the pretrained pLM and attaching the task head
- Resolving odd dependency issues that inevitably arises
- Implementing PEFT by directly modifying the pLM architecture

## Standardization of existing transformer models

Fortunately, these problems boil down to mainly three common problems:

- **Setting up the model** (having to install/clone, full environment or container setup for each model)
- **Using model-specific syntax** (tokenizer, attention mask, padding, etc)
- **Wiring with prediction head and fine-tuning**

Given that these are repeated problems, people have developed a framework that can be used to remove much of the pain from them. **That is what Hugginface does**.

Huggingface helps solve these issues by providing a unified interface to access and subclass various pretrained transformer models. Through the `transformer` library we can use most of existing pLMs through a unified syntax. Let’s quickly see how it helps with each of the three pain points before we move onto practical implementation of fine-tuning in the next post.

- **Setting up the model → Importing using transformer library**
    - Huggingface provides a single, simple API (`AutoModel.from_pretrained`) to load pretrained models and tokenizer from the Hugging Face Hub, automatically handling the model architectures and pretrained weights. There is no need to install or clone each model. For example, generating ESM2 embedding for proteins looks like this.
    
    ```python
    from transformers import AutoModel, AutoTokenizer # These libraries handle loading tokenizer and model from name
    
    # Load ESM2 tokenizer and pretrained 650M-parameter model
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    
    # Generate embedding
    inputs = tokenizer("MKTAYIAKQRQISFVKSHFSRQDILDLIC", return_tensors="pt") #Returns BatchEncoding object with input_ids and attention_mask
    outputs = model(**inputs) #Returns ModelOutput class with loss, logits, hidden_states, attentions
    embeddings = outputs.last_hidden_state  # shape: [batch_size, sequence_length, hidden_dim]
    ```
    
    If we want to use a different model, ProBert:
    
    ```python
    # Load ProtBert tokenizer and pretrained ProBert
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
    model = AutoModel.from_pretrained("Rostlab/prot_bert")
    
    # ProtBert expects spaces between amino acids
    sequence = "MKTAYIAKQRQISFVKSHFSRQDILDLIC"
    sequence_spaced = ' '.join(list(sequence)) #For ProBert, sequence must be formatted M K T A ...
    
    # Generate embedding
    inputs = tokenizer(sequence_spaced, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    ```
    
    This second example shows that we still need to understand some quirks about the models we want to use. Here, ProtBert expects for inputs a single space between amino acids. 
    
    Also, huggingface does not directly handle separate environments or conflicting dependency versions for each model. So if there’s a clash in libraries required by two models, this will require careful virtual environment setup. However, as long as common dependency versions for packages are compatible across multiple models, which is often the case, then multiple transformer models can be used in the same environment without needing separate isolated setups.
    
- **Using model-specific syntax → Using unified syntax**
    - Huggingface standardizes input/output data structures (tokenized inputs, attention masks, positional encodings and `ModelOutput` objects), so methods like `.encode()`, `.decode()`, `.forward()`, and `.generate()` work the same way across different transformer models.
- **Wiring with prediction head and fine-tuning → Simplified by unified API**
    - Huggingface provides built-in utilities (`Trainer`, `TrainingArguments`) that standardize training loops, logging, evaluation, hyperparameter tuning, and distributed training.
    - The `PEFT` (for parameter-efficient fine-tuning) library provides ways to implement fine-tuning techniques like LoRA with any model from the `PreTrainedModel` class.

These second and third points will become clearer when we look at practical implementation of fine-tuning pLMs.

# **Workflow for fine-tuning**

Before getting into the code, let’s conceptually break down what a full parameter pLM fine-tuning for a prediction task requires.

1. Define the prediction head (classification/regression) that uses pLM embedding as input.
2. Define the main model that wires the pLM and the prediction head together. Initializes the pLM with pretrained weights.
3. Prepare labeled datasets for supervised training as well as validation and test.
4. Define optimizer and trainer.
5. Pipeline to bring everything together.


Along the way, we will need to pay attention to correctly handling tokenization, attention mask, and padding/truncation. But Huggingface framework will help us with these too.

## Setting the goals for what the code should do

In this post I will share the code for pLM fine-tuning. **The key focus, which was most of the challenge, was to write a nice modular code that:**
* can be used with various pLMs as plug-and-play
* provides some template for simple prediction head, but can also work with other custom prediction head models by allowing passing of additional arguments. 

Although huggingface provides a convenient interface, there are still some model-specific quirks that made this somewhat tricky. For example, to enable various prediction heads to work as plug-and-play, the following issues must be considered.
- Models may make residue-level or protein-level prediction, and classification or regression. We need to use appropriate loss function for each case. For example:
    - Residue-level Classification
        - Example: does each residue belong to intrinsically disordered region?
        - Loss function: cross entropy loss summed over residues (exclude padded residues)
    - Residue-level Regression
        - Example: per-residue evolutionary mutability
        - Loss function: MSE loss, MAE loss, etc. summed over residues (exclude padded residues)
    - Protein-level Classification
        - Example: classify a protein as binder or non-binder to a given target
        - Loss: cross entropy loss on sequence-level logits
    - Protein-level Regression
        - Example: prediction melting temperature Tm
        - Loss: MSE loss, MAE loss, etc at sequence level

* For making a protein-level prediction, there are different ways of aggregating the embeddings across the residues. For example, some BERT-based models like ProtBert may have the special cls token that can be used for classification. User may choose to use it, or ignore that and take the mean of the embeddings across the residues.

Moreover, the pLM themselves will also have model-specific quirks:
* Data pre-processing steps needs to be correctly handled. As pointed in Part 1, the ProtBert model requires uppercase amino acids that are separated by spaces.
* Each pLM will have certain model-specific attributes in the model architecture. For example, T5-based models like `ProtT5` has `self.shared` layer that implements vocabulary encoding. The name `shared` comes from the fact that it is a shared matrix by the encoder and decoder. Naturally, encoder-only models like `ESM2` will not have this layer. If we want a modular class for our main model that enables plug-and-play with different pLMs, we should avoid referencing specific attributes like this and only use attributes that are universal for the `PreTrainedModel` class in transformer library.

When we look at [RSchmirler et al. repo](https://github.com/RSchmirler/data-repo_plm-finetune-eval), they defined separate classes for fine-tuning different pLM models (e.g. T5EncoderForSimpleSequenceClassification and load_T5_model for ProtT5, load_esm_model for ESM2, etc) and also for different tasks (e.g. T5EncoderForTokenClassificaion and T5EncoderForSimpleSequenceClassification are defined separately although most of the functionality is redundant). While this works fine for the scope of their study, it would be nice to have a more modular framework.

# **Next Steps**
In the next post, I will go through the steps mentioned above for pLM fine-tuning with code examples