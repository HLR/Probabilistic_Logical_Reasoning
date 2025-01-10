# Teaching Probabilistic Logical Reasoning to Transformers (EACL 2024 Findings)

## Abstract

We evaluate the capability of transformer-based language models in uncertain text that includes uncertain rules of reasoning. We cover both Pre-trained Language Models (PLMs) and the newer generative Large Language Models (LLMs). Our evaluation results show that both generations of language models struggle with reasoning over uncertain text. We propose a novel end-to-end fine-tuning approach, Probabilistic Constraint Training (PCT), that utilizes probabilistic logical rules as constraints in the fine-tuning phase without relying on these rules in the inference stage. To assess the effectiveness of PCT, we utilize the related corpora and, additionally, create a new and more challenging benchmark that, unlike the previous ones, uses instance-specific rules. Our study demonstrates that PCT improves the transformer-based language model's intrinsic reasoning and the explainability of the probabilistic logical reasoning process. Furthermore, PCT equips these models to effectively handle novel situations, including higher reasoning depth, new domains, and complex probabilistic structures.

## Key Contributions

1. **Probabilistic Constraint Training (PCT)**: We propose a new approach, Probabilistic Constraint Training (PCT), that explicitly imposes probabilistic reasoning rules during PLM fine-tuning. This approach provides an effective level of abstraction to the models to generalize and transfer reasoning under uncertainty to new domains and more complex depths of reasoning.

2. **New Evaluation Benchmark**: We develop a novel evaluation benchmark for probabilistic reasoning over text with context-specific uncertain rules whose probabilities can not be captured from the training data and must be extracted from the text.

3. **Comparative Experiments with LLMs**: We conduct thorough experiments comparing our constraint-based fine-tuning approach with LLMs and show the superiority of our technique.

## Dataset and Methodology

We utilized RuleBERT and our new RuleTaker-pro datasets, focusing on probabilistic logical inference from linguistic expressions of rules and facts. Our approach involves converting reasoning steps into equality constraints, ensuring output consistency during training.

## Repository Contents

- [RuleBERT Code](https://github.com/HLR/Probabilistic_Logical_Reasoning/tree/main/RuleBERT) Source code for implementing PCT In RuleBERT. 
- [RuleTaker-pro Code](https://github.com/HLR/Probabilistic_Logical_Reasoning/tree/main/RuleTaker-pro) Source code for implementing PCT In RuleTaker-pro.
- [RuleTaker-pro Dataset](https://github.com/HLR/Probabilistic_Logical_Reasoning/tree/main/RuleTaker-pro/Dataset/)  RuleTaker-pro dataset files.

## Getting Started

To use our code and datasets, please follow the instructions in the [RuleBERT Code](https://github.com/HLR/Probabilistic_Logical_Reasoning/tree/main/RuleBERT) and [RuleTaker-pro Code](https://github.com/HLR/Probabilistic_Logical_Reasoning/tree/main/RuleTaker-pro) repositories.

## Citing Our Work

If you use our work in your research, please cite our paper as follows:
```bibtex
@inproceedings{nafar-etal-2024-teaching,
    title = "Teaching Probabilistic Logical Reasoning to Transformers",
    author = "Nafar, Aliakbar  and
      Venable, K. Brent  and
      Kordjamshidi, Parisa",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2024",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-eacl.112/",
    pages = "1615--1632",
    abstract = "In this paper, we evaluate the capability of transformer-based language models in making inferences over uncertain text that includes uncertain rules of reasoning. We cover both Pre-trained Language Models (PLMs) and generative Large Language Models (LLMs). Our evaluation results show that both generations of language models struggle with reasoning over uncertain text. We propose a novel end-to-end fine-tuning approach, Probabilistic Constraint Training (PCT), that utilizes probabilistic logical rules as constraints in the fine-tuning phase without relying on these rules in the inference stage. To assess the effectiveness of PCT, we utilize the related corpora and, additionally, create a new and more challenging benchmark that, unlike the previous ones, uses instance-specific rules. Our study demonstrates that PCT improves the transformer-based language model`s intrinsic reasoning and makes their probabilistic logical reasoning process more explicit and explainable. Furthermore, PCT equips these models to effectively handle novel situations, including higher reasoning depth, new domains, and complex probabilistic structures."
}
