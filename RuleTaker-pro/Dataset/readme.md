# Dataset Description

## Overview

This dataset presents a collection of probablistic logical reasoning problems, each encapsulated in a single row. 

RuleTaker-pro is created to address some of the shortcomings of the RuleBERT dataset. RuleBERT is built using about 100 rules with fixed probabilities that are applied to many examples in the dataset. The probabilities of these rules are extracted from an external source and remain constant for all examples in the dataset. However, we want a dataset with example-specific rules to make the required reasoning more realistic. 

## Dataset Structure

Each entry in the dataset is comprised of the following columns:

- `context`: A textual description providing the background and rules that apply to the scenario. This includes conditional statements and descriptions of various entities.
- `question`: A specific question about the scenario that requires probablistic logical deduction to answer.
- `label`: A boolean value (`True` or `False`) indicating the correctness of the proposition presented in the question.
- `proba`: A numerical value (ranging from 0 to 1) representing the probablity question being true.
- `depth`: A numerical indicator of the logical depth required to answer the question, where a higher value suggests a more complex reasoning process is needed.
- `complex`: A boolean value indicating whether the question involves complex reasoning beyond basic linear logical deduction.

## Adverbs

The adverbs in the context correspond to the following probabilities:

- **Always**: This adverb corresponds to a scenario happening with a certainty or a probability of 100%. It signifies that the event in question will occur every time without fail.

- **Usually**: This adverb indicates that the event in question happens with a probability of 90%. It means that there is a high likelihood of the event occurring, though not with absolute certainty.

- **Normally**: This adverb suggests that the event has an 80% probability of occurring. It implies that under regular or standard conditions, the event is quite likely to happen.

- **Often**: This adverb is associated with a 65% probability of the event happening. It suggests that the event occurs more times than not, but it is not as frequent as "usually" or "normally".

- **Sometimes**: This adverb signifies a balanced probability of 50%, indicating that the event is just as likely to happen as it is not to. It reflects an even chance of occurrence.

- **Occasionally**: With a probability of 30%, this adverb points to events that happen less frequently. It indicates a lower likelihood of occurrence, suggesting that while the event can happen, it does so infrequently.

- **Seldom**: This adverb corresponds to a 15% probability, indicating that the event rarely happens. It suggests a significantly low likelihood of the event occurring.

- **Never**: This adverb is associated with a 0% probability, indicating that the event does not happen at all. It represents an absolute certainty that the event will not occur.

### Removing Adverbs
If you wish the replace the adverbs with numerical values simply remove the `--adverb` from the `train.py` files.
if you wish to use the dataset outside of our framwork and replace the adverbs, include the following code after loading the dataset:

```python
adverb_replacements = {
    "always": "with a probability of 100%",
    "usually": "with a probability of 90%",
    "normally": "with a probability of 80%",
    "often": "with a probability of 65%",
    "sometimes": "with a probability of 50%",
    "occasionally": "with a probability of 30%",
    "seldom": "with a probability of 15%",
    "never": "with a probability of 0%"
}
dataset=pd.read_csv(f"dataset/d{depth}/{split}-pro.csv")
dataset['context'] = dataset['context'].replace(adverb_replacements, regex=True)
```
