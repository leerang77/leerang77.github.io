---
layout: post
title: Fine-tuning protein language model with Huggingface (Part 2)
date: 2025-07-27 15:09:00
description: Practical workflow for fine-tuning with Huggingface
tags: protein-language-model
categories: Bio-ML
featured: false
related_posts: false
---

# **Intro**
In Part 1 of the post, I went over motivation and intuition for fine-tuning pLMs, distinguished task-adaptive fine-tuning from domain-adaptive pretraining, introduced parameter-efficient fine-tuning, and briefly introduced Huggingface. This post will be go more in-depth on examples of fine-tuning code with Huggingface libraries. Specifically, we will cover:
1. **Practical examples and workflow of fine-tuning code with Huggingface libraries**
2. **Parameter-efficient fine-tuning with Hugginface PEFT library**

# **Code for full parameter fine-tuning**
Below, I will show the code for four steps necessary for pLM fine-tuning, using Huggingface libraries.

1. Defining the prediction head to be used with pLM
2. Defining the main model that wires pLM and task model together
3. Defining the data module
4. Defining the optimizer and trainer

## **1. Defining the prediction head to be used with pLM**
For task-adaptive fine-tuning, we need a prediction head. Let’s define `MLPCHead` that can handle both residue-level and protein-level prediction tasks, and both embedding-mean and cls-token pooling strategies if protein-level prediction is used. The MLP architecture is a simple template here, and any other prediction task model (e.g. graph-based models) can be defined similarly. 

```python
"""
Example prediction task model with MLP architecture
"""
import torch.nn as nn

class MLPHead(nn.Module):
    """Modular MLP head with configurable pooling method.
    Supports per-protein (mean or CLS) or per-residue classification."""

    def __init__(
        self,
        input_dim: int,         # Should match the pLM embedding dimension
        hidden_dim: int,        # Hidden layer dimensions
        output_dim: int,        # Number of classes (1 if regression)
        num_hidden_layers: int = 1, # Variable number of hidden layers in MLP
        dropout_rate: float = 0.1,  # Dropout rate
        classification_mode: str = "protein", # 'protein' or 'residue'
        pooling_strategy: str = "mean",       # 'mean' or 'cls' when protein-level
    ):
        """
        Initializes the MLP prediction head.
        """
        super().__init__()
        # Define classification mode
        assert classification_mode in (
            "protein",
            "residue",
        ), "classification_mode must be 'protein' or 'residue'"
        # Define pooling strategy
        if classification_mode=="protein":
		        assert pooling_strategy in (
		            "mean",
		            "cls",
		        ), "pooling_strategy must be 'mean' or 'cls'"
        self.classification_mode = classification_mode
        self.pooling_strategy = pooling_strategy

        # Define the architecture with the input num_hidden_layers
        self.output_dim = output_dim
        dims = [input_dim] + [hidden_dim] * num_hidden_layers + [output_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.mlp = nn.Sequential(*layers)


    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward pass for the MLP head.
        """
        if self.classification_mode == "protein": # Protein-level prediction
            if self.pooling_strategy == "cls": # BERT-style 'cls' token
                x = hidden_states[:, 0, :]
            else: # Use mean of embeddings 
                if attention_mask is not None: # Exclude padding from the mean
                    mask = attention_mask.unsqueeze(-1)
                    masked = hidden_states * mask
                    sum_hidden = masked.sum(dim=1)
                    lengths = mask.sum(dim=1).clamp(min=1)
                    x = sum_hidden / lengths
                else:
                    x = hidden_states.mean(dim=1)
            return self.mlp(x)

        else: # Residue-level prediction
            B, L, D = hidden_states.shape
            flat = hidden_states.view(B * L, D)
            logits_flat = self.mlp(flat)
            return logits_flat.view(B, L, self.output_dim)
```

## **2. Defining the main model**
Now that we have the prediction head, we need to define a model class that wires the prediction head and pLM together so that we can backpropagate through both during the supervised training. In the below example, we define a model that can be used with both protein-level and residue-level prediction, and both classification and regression tasks. As mentioned in the previous post, each of these cases require different loss functions. To help with this, let's first define `TaskType` Enum class.  

```python
class TaskType(Enum):
    """
    Enum representing different task types for the prediction head.
    """
    SEQ_CLASSIFICATION   = "seq_classification"
    SEQ_REGRESSION       = "seq_regression"
    TOKEN_CLASSIFICATION = "token_classification"
    TOKEN_REGRESSION     = "token_regression"
```

Now let’s define `PLMTaskModel` class which is our main model. It first uses `AutoModel` to load the specified pre-trained pLM and assign it to `self.backbone`. The `forward()` method first extracts embedding using `self.backbone` and then calls the prediction head using the `last hidden_states`, along with other `kwargs` required by the prediction task model. For example, if the prediciton head is a graph-based model, the graphs may be passed as additional arguments.

```python
from typing import Optional, Callable, Any

import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    PreTrainedModel,
    SequenceClassifierOutput,
)

def PLMTaskModel(PreTrainedModel):
    """General model for sequence/token classification and regression."""

    def __init__(
        self,
        task_type: TaskType,
        backbone_name: str,
        head: nn.Module,
    ):
        """
        Initializes the PLMTaskModel with a backbone model, a head for task-specific processing,
        and an optional preprocessing function.
        
        Args:
            task_type (TaskType): Type of the task (e.g., SEQ_CLASSIFICATION, TOKEN_CLASSIFICATION).
            backbone_name (str): Name of the pretrained model backbone.
            head (nn.Module): Task-specific head to be used on top of the backbone.
            preprocess_fn (Callable[[str], str]): Function to preprocess input sequences.
        """
        # Load the config and backbone
        config = AutoConfig.from_pretrained(backbone_name)
        backbone = AutoModel.from_pretrained(backbone_name, config=config)
        
        # Call the PretrainedModel constructor
        super().__init__(config)
        
        # Attach the modules
        self.backbone = backbone # Assigns the pretrained weights
        self.head = head # Assigns the prediction head
        self.task_type = task_type
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **head_args: Any,
    ) -> SequenceClassifierOutput:
        """
        Forward pass for the PLMTaskModel.
        
        Args:
            input_ids (torch.LongTensor): Input token IDs.
            attention_mask (Optional[torch.FloatTensor]): Attention mask.
            labels (Optional[torch.LongTensor]): Labels for classification tasks.
            **head_args (Any): Additional arguments for the head.
        Returns:
            SequenceClassifierOutput: Output of the model including logits and loss if labels are provided.
        """
        # Compute embedding
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.head(hidden_states, attention_mask=attention_mask, **head_args)
        
        # Compute loss
        loss = None
        if labels is not None:
            if self.task_type == TaskType.TOKEN_REGRESSION:
                # logits: (batch, seq_len, 1) → squeeze
                preds = logits.squeeze(-1)                    # (batch, seq_len)
                # build a mask of the real (non-pad) tokens
                mask  = attention_mask.to(preds.dtype)        # 1.0 for real tokens, 0.0 for pads
    
                # compute squared error only on real tokens
                se    = (preds - labels.float()) ** 2         # (batch, seq_len)
                loss  = (se * mask).sum() / mask.sum()        # mean over real positions
    
            elif self.task_type == TaskType.TOKEN_CLASSIFICATION:
                    # logits: (batch, seq_len, num_labels)
                loss = nn.CrossEntropyLoss(ignore_index=-100)(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
    
            elif self.task_type == TaskType.SEQ_REGRESSION:
                    # logits: (batch, 1)
                loss = nn.MSELoss()(logits.squeeze(-1), labels.float())
    
            else:  # SEQ_CLASSIFICATION
                    # logits: (batch, num_labels)
                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
     
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states   if output_hidden_states else None,
            attentions=outputs.attentions         if output_attentions    else None,
        )
```
Now we have our model. Next, we define our data module.

## **3. Defining the data module**
We define the data module that loads, preprocesses, and tokenizes the data. To make it modular and compatible with various pLMs, we use the huggingface `AutoTokenizer` as input so that the appropriate model-specific tokenizer can be passed. It also use `preprocess_fn` as an optional input to handle any model-specific quirk (e.g. for ProtBert model, a function that adds a space between amino acids) inside the data module. We return the training, validation and optional test datasets as Huggingface `Dataset` objects, within a single `DatasetDict` object.

```python
from typing import Optional, Callable, List

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict

class ProteinDataModule:
    """
    Data module for protein sequence datasets, handling loading, preprocessing,
    and tokenization using Hugging Face's datasets library.
    This module supports training, validation, and optional test datasets.
    """
    def __init__(
        self,
        train_file: str,
        val_file: str,
        tokenizer: AutoTokenizer,
        preprocess_fn: Optional[Callable[[str], str]] = None,
        max_length: int = 1024,
        test_file: Optional[str] = None,
        optional_features: Optional[List[str]] = None,

    ) -> DatasetDict:
        """
        Initializes the ProteinDataModule with training and validation files,
        a tokenizer, and optional preprocessing function and optional test file.
        Args:
            train_file (str): Path to the training dataset file.
            val_file (str): Path to the validation dataset file.
            tokenizer (AutoTokenizer): Tokenizer for processing sequences.
            preprocess_fn (Optional[Callable[[str], str]]): Function to preprocess sequences.
            max_length (int): Maximum length for tokenized sequences.
            test_file (Optional[str]): Path to the test dataset file, if available.
            optional_features (Optional[List[str]]): Additional features to include in the dataset.
        """
        self.tokenizer = tokenizer
        self.preprocess_fn = preprocess_fn
        self.max_length = max_length
        self.optional_features = optional_features if optional_features else []
        files = {"train": train_file, "validation": val_file}
        if test_file:
            files["test"] = test_file
        raw = load_dataset("csv", data_files=files)

        def preprocess(examples):
            """
            Preprocesses the input examples by applying the preprocessing function
            and tokenizing the sequences.
            """
            seqs = examples["sequence"]
            if self.preprocess_fn: # Optional preprocessing step (e.g. Add space for ProtBert)
                seqs = [self.preprocess_fn(s) for s in seqs]
            tokenized = self.tokenizer(
                seqs,
                truncation=True,
                max_length=self.max_length,
            )
            if "label" in examples:
                tokenized["labels"] = examples["label"]
            for key in self.optional_features: # Optional keys for additional features (e.g. graph)
                if key in examples:
                    tokenized[key] = examples[key]
            return tokenized

        self.datasets = raw.map(preprocess, batched=True)

    def get_datasets(self) -> DatasetDict[str, Dataset]:
        """
        Returns the processed datasets for training, validation, and optional test.
        Returns:
            DatasetDict[str, Dataset]: Dictionary containing the processed datasets.
        """
        return self.datasets
```

Here's an example of loading and preprocessing a dataset:
```python
"""
    Example for creating dataset to be used with ProtBert
"""

# 1. Define a preprocessing function that upper-cases and spaces out residues
def ProtBert_preprocess(seq: str) -> str:
    """
    Turn a contiguous amino-acid string into uppercase
    letters separated by spaces.
    E.g. "mkta" → "M K T A"
    """
    seq = seq.strip().upper()
    return " ".join(list(seq))

# 2. Load ProtBert’s tokenizer (it expects space-separated amino acids)
tokenizer = AutoTokenizer.from_pretrained(
    "Rostlab/prot_bert",
    do_lower_case=False,
)

# 3. Instantiate your ProteinDataModule, pointing at CSVs with a "sequence" column
ProtBert_data_module = ProteinDataModule(
    train_file="data/train.csv",
    val_file="data/val.csv",
    tokenizer=tokenizer,
    preprocess_fn=ProtBert_preprocess,
    max_length=1024,
    test_file="data/test.csv",     # optional
)

# 4. Get the tokenized DatasetDict
ProtBert_datasets: DatasetDict = data_module.get_datasets()
```


## **4. Defining the optimizer and trainer**
Now that we have defined the model and the dataset, we now need to define optimizer and trainer, and optionally a scheduler for the optimizer. While we can do this with Pytorch, once again Hugginface provides a `Trainer` class that simplifies the process. The `Trainer` class in addition enables simplified workflow for distributed training and mixed-precision handling as well.

`Trainer` class actually by default implements AdamW optimizer and a linear scheduler with warmup and decay, so there's no need to explicitly define them. It is highly customizable through the use of `TrainingArguments` class that is supplied as input to the trainer. [Trainer documentation](https://huggingface.co/docs/transformers/en/main_classes/trainer) from Huggingface shows there are **118** parameters that can be passed to TrainingArguments.

In this example, we will assume that we have written a metrics module with get_compute_metrics_fn that returns appropriate metrics function given the model task type. We use huggingface `DataCollatorWithPadding` or `DataCollatorForTokenClassificaiton` to implement per-batch dynamic padding to the length of the longest sequence in each batch. If we have  We then use huggingface `Trainer` module, with `TrainingArguments` definition. By doing so, we can use pre-defined `trainer.train()` and `trainer.evaluate()` methods to simplify training.

```python
"""
Trainer Module
"""
from typing import Optional, Dict
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
)
# from metrics import get_compute_metrics_fn
from plft.metrics import get_compute_metrics_fn
from plft.model import PLMTaskModel
from plft.config import TaskType

class ProteinTaskTrainer:
    """
    Trainer for protein sequence tasks, handling training and evaluation
    using Hugging Face's Trainer API.
    """
    def __init__(
        self,
        model: PLMTaskModel,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        test_dataset: Optional[Dataset],
        tokenizer: AutoTokenizer,
        output_dir: str,
        num_train_epochs: int = 100,
        per_device_train_batch_size: int = 8,
        learning_rate: float = 1e-4,
    ):
        """
        Initializes the ProteinTaskTrainer with model, datasets, tokenizer, and training parameters.
        """
        self.model         = model
        self.train_dataset = train_dataset
        self.eval_dataset  = eval_dataset
        self.test_dataset  = test_dataset

        # pick the right collator:
        # for residue-classification, pad labels to -100 so CrossEntropyLoss(ignore_index=-100) skips them
        # for everything else (seq-classification, seq-/residue-regression), plain padding is sufficient
        if self.model.task_type == TaskType.TOKEN_CLASSIFICATION:
            self.data_collator = DataCollatorForTokenClassification(
                tokenizer,
                label_pad_token_id=-100,
            )
        else:
            self.data_collator = DataCollatorWithPadding(tokenizer)

        compute_metrics = get_compute_metrics_fn(self.model.task_type) # Get the appropriate metrics function based on task type

        # Initialize the Trainer
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            tokenizer=tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
        )

    def train(self):
        """
        Train the model using the huggingface Trainer module.
        """
        return self.trainer.train()

    def evaluate(self, split: str = "validation") -> Dict[str, float]:
        """
        Evaluate the model on the specified dataset split (train, validation, or test).
        Args:
            split (str): The dataset split to evaluate on. Can be "train", "validation", or "test".
        Returns:
            Dict[str, float]: A dictionary containing evaluation metrics.
        """
        if split == "train":
            ds = self.train_dataset
        elif split == "validation":
            ds = self.eval_dataset
        elif split == "test":
            if self.test_dataset is None:
                raise ValueError("No test dataset provided.")
            ds = self.test_dataset
        else:
            raise ValueError(f"Unknown split: {split}")
        return self.trainer.evaluate(eval_dataset=ds)
```

# **Parameter-efficient fine tuning with PEFT library**
Now that we have defined the model, datamodule, and trainer, we are almost ready for training. But there is one thing still missing: implementing parameter-efficient fine-tuning. In the previous post we briefly mentioned what it is, and that we will focus on Low Rank Adaptation (LoRA) method. Huggingface PEFT library makes the implementation of LoRA incredibly simple, so let's look at the code first. 

```python
from peft import LoraConfig, get_peft_model

base_model = PLMTaskModel(...) 

# Create a LoRA config.
lora_config = LoraConfig(
    r=8,                         # LoRA rank
    lora_alpha=32,               # scaling
    target_modules=["q_proj",    # which modules to inject into
                    "v_proj"],  
    dropout=0.05,                # optional
    bias="none",
    task_type="SEQ_CLS"          # one of: "SEQ_CLS", "SEQ_REG", "TOKEN_CLS", "TOKEN_REG"
)

# Wrap the model
peft_model = get_peft_model(base_model, lora_config)
```

Then, use peft_model instead of base_model in the rest of the code. That's it! This code updates the query and value projection matrices (i.e. `q_proj` and `v_proj`) using rank-8 matrices. A standard attention head computes

$$
Q = X\,W_Q
$$

$$
V = X\,W_V
$$

with $W_Q, W_V\in\mathbb R^{d\times d_k}$, where $d_k$ is the dimension of attention head. With LoRA, instead of learning a full update to $W_Q$, we are introducing

* $A_Q\in\mathbb R^{d\times r}$
* $B_Q\in\mathbb R^{r\times d_k}$

where $r \ll \min(d, d_k)$ is the rank.
Using these, we modify the query and value as:

$$
\begin{aligned}
Q &= X\bigl(W_Q + \tfrac{\alpha}{r}B_QA_Q^T\bigr),\\
\end{aligned}
$$

Similarly, $W_V$ is updated as:

$$
\begin{aligned}
V &= X\bigl(W_V + \tfrac{\alpha}{r}B_VA_V^T\bigr),\\
\end{aligned}
$$

The hyperparameter $\alpha$ scales the adapter’s effect.

During the training, only $A_Q, B_Q, A_V, B_V$ for each attention head are updated. All other weight matrices such as keys, output projections, feed-forward layers, etc. stay frozen during the training.