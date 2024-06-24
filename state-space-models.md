# State Space Models (SSM) - Motivation & Mamba

Mamba Paper: https://arxiv.org/pdf/2312.00752

In this blog post I'll discuss SSM motivation, explore differences with RNNs/LSTMs and Transformers.

## RNN:
* O(n) in memory during training (store gradients for each state)
* O(1) in memory during inference
* O(n) in compute during training
* O(n) in compute during inference
* Bad for training - we cannot parallelize w.r.t tokens during training. In given input sequence, we know its corresponding output sequence, but we still must compute 1 by 1. (compare with transformers which can parallelize w.r.t tokens)
Vanishing/exploding gradients

## Transformers
* O(n^2) in memory and computation during training. Terrible. 
* O(N) in memory and computation (think how many dot products we do, its o(n)) during inference using kv-cache 
We can easily parallelize during training the whole sequence computation.

SSM offers the ideal combination:
1) Easy training parallelization (like Transformers) for handling long sequences in memory (like RNNs).
2) Constant computation/memory for each token inference.



