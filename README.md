# NLP-Selection-Camp-BERT-Implementation
A compact, educational PyTorch implementation of a simplified BERT (encoder-only Transformer) trained on WikiText-2 with Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).
Designed for small-scale experiments (not production) — suitable for notebooks and limited compute.

## Features
- Encoder-only Transformer (Multi-head Self Attention + Feedforward + LayerNorm + Residual)
- WordPiece tokenizer (`bert-base-uncased`)
- Uses [CLS] and [SEP] tokens
- Segment embeddings (sentence A vs sentence B) for NSP
- MLM head + NSP head implemented manually
- Weight tying between embedding matrix and MLM output layer
- WikiText-2 dataset (HuggingFace Datasets)

## Important information relating to the project
- I have used the Pretrained Bert Tokenizer ('bert-base-uncaed') for tokenizing the input sentences.
- I am training for only 3 epochs due to limited resources.
- NSP Acc. is nearly 0.54 but will improve on training further.

## Installation
For simplicity, I have created **BERT.ipynb** which contains the code and can be run locally or online. The trained model is saved to **mini_bert_weights.pt** can can be loaded directly for use/ testing without retraining.
