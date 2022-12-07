# Good, Fast, Cheap: Pick One (Transfer Learning for Multimodal Audio Retrieval)
**A Final Course Project for Georgia Tech's CS7643/4643 Deep Learning course**\
Authors: Christian Clark, Nikhil Ravi Krishnan, Walter Becerra

This project focuses on audio retrieval from natural language captions (and vice versa). You can download and run this code to replicate the work from our course project report, found in this main folder as a pdf. All models run in this paper used the random seed "7643".

This repo contains the code for a contrastive learning model with 3 different audio/image encoders and RoBERTa, aligned in a joint space through a linear layer for each encoder. The code for these models is located in the clip.py file.

## How to Use this Code
To train a model:
```python main.py --config="./path/to/config.yaml" --mode="train"```

To evaluate a model:
```python main.py --config="./path/to/config.yaml" --mode="evaluate"```

To see more information about the .yaml file format, see the "configs" folder.
