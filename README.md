# AttentionIsAllYouNeed
This repository contains a full PyTorch implementation of the original encoder-decoder Transformer architecture from ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762) paper by Vaswani et al. If you’d like a step-by-step walkthrough of how the Transformer works and how each part fits together, check out my accompanying [Medium article](https://medium.com/@jshim1213/dedication-is-all-we-need-recreating-the-original-transformer-2c29298eae63)!

## Installation Guide
1. Set up the environment using conda or install locally:
```
conda create -n transformers python==3.10.12
conda activate transformers
pip install -r requirements.txt
```

2. clone the repo

## Training
The config currently in `train.py` mimics the original paper's configs for the base model, and the commented out config can be used to train on a subset of data points.
```
python train.py
```

## Evaluation
Edit the `eval.py` to load in your model checkpoint and it will output the corresponding BLEU score and some sample responses for visual comparison.

## Sample Results
Loaded checkpoint from epoch 14

Training loss was: 2.7890

BLEU score: **15.34**

Evaluated on -1 samples


### Sample Translations

**Source**: Gutach: Noch mehr Sicherheit für Fußgänger

**Reference**: Gutach: Increased safety for pedestrians

**Hypothesis**: More safety: pedestrians
____

**Source**: Sie stehen keine 100 Meter voneinander entfernt: Am Dienstag ist in Gutach die neue B 33-Fußgängerampel am Dorfparkplatz in Betrieb genommen worden - in Sichtweite der älteren Rathaus
ampel.

**Reference**: They are not even 100 metres apart: On Tuesday, the new B 33 pedestrian lights in Dorfparkplatz in Gutach became operational - within view of the existing Town Hall traffic lights.

**Hypothesis**: They are not visible from each other: on Tuesday the new town park in Bampachel was put into operation at the older village square in the village park - 33 meters away.

____
**Source**: Zwei Anlagen so nah beieinander: Absicht oder Schildbürgerstreich?

**Reference**: Two sets of lights so close to one another: intentional or just a silly error?

**Hypothesis**: Two systems: so close to each other or to the citizen?

____
**Source**: Diese Frage hat Gutachs Bürgermeister gestern klar beantwortet.

**Reference**: Yesterday, Gutacht's Mayor gave a clear answer to this question.

**Hypothesis**: This question was clearly answered yesterday by Mayor Gutsach.
____

**Source**: "Die Rathausampel ist damals installiert worden, weil diese den Schulweg sichert", erläuterte Eckert gestern.

**Reference**: "At the time, the Town Hall traffic lights were installed because this was a school route," explained Eckert yesterday.

**Hypothesis**: "The corner of the town was installed yesterday, because this corner ensures the school path," explains.
