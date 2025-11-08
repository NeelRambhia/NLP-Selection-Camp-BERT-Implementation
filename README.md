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
