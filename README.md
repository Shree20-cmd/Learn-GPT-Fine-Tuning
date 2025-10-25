# Learn-GPT-Fine-Tuning

GPT (Generative Pre-trained Transformer) is an autoregressive language model, meaning it predicts the next token in a sequence given all the previous tokens.

P(x1​,x2​,…,xn​) = t(1to n)∏​P(xt​∣x1​,x2​,…,xt−1​)  , the model learns to model the joint distribution over all words (or tokens) in a text.

## The Core Difference: GPT vs. BERT

| Aspect         | GPT                                                        | BERT                                                     |
| -------------- | ---------------------------------------------------------- | -------------------------------------------------------- |
| Objective      | Predict **next token** (autoregressive)                    | Predict **masked tokens** (autoencoding / bidirectional) |
| Context        | **Left-to-right** (causal attention)                       | **Bidirectional** (can look at both sides)               |
| Architecture   | **Decoder-only Transformer**                               | **Encoder-only Transformer**                             |
| Training       | Text generation objective                                  | Masked language modeling + Next sentence prediction      |
| Usage          | Generation, completion, dialogue, summarization, reasoning | Understanding tasks (classification, QA, embeddings)     |
| Attention Mask | Causal mask (prevents looking at future tokens)            | Full attention (can see all tokens)                      |

BERT is like a deep reader,
GPT is like a storyteller.

## GPT Architecture
GPT uses the Transformer Decoder Block, repeated N times.
Each token is converted to an embedding vector: xt​=E(wt​)+P(t)  , token embedding and positional encoding

Self-Attention Mechanism

Each layer uses Multi-Head Self-Attention (MHSA).
Each head transforms inputs into queries (Q), keys (K), and values (V).

For each head: Q=XWQ​,K=XWK​,V=XWV​ 

A = Softmax(​QKT/(dh​**1/2)​+M) V , M is the mask (GPT uses a causal mask so token t only attends to ≤ t).

Then all heads are concatenated and linearly projected: MHSA(X)=[head1​,…,headh​]WO​ 

Feed-Forward Network (FFN)

FFN(x)=GELU(xW1​+b1​)W2​+b2​

Residual + LayerNorm Connections

X′=LayerNorm(X+MHSA(X))
Y=LayerNorm(X′+FFN(X′))

## Stacking Layers

GPT stacks N such decoder blocks (e.g., 12 for GPT-2 small, 96+ for GPT-4).
Each layer captures higher-level semantics.

## Output Projection

Finally, we project the final hidden state to the vocabulary space:

P(xt+1​∣x≤t​)=Softmax(ht​*WET​)

## Training Objective

The model minimizes cross-entropy loss over next-token predictions:

L= −t=1∑n​logP(xt​∣x<t​)

## General Tips for Hyperparameter Tuning

Start Small: The code’s defaults (lr=5e-5, batch_size=4, num_epochs=3, max_length=128) are good for testing on 1,000 examples. Verify they work before scaling up.
Monitor Loss: The code prints average loss per epoch. Expect it to start at ~3–4 and drop to ~1–2. If it doesn’t decrease, reduce lr or check data formatting.
Hardware Constraints:

GPU Memory: For gpt2 (124M parameters), 8GB VRAM handles batch_size=4, max_length=128. For larger models (e.g., gpt2-medium), reduce batch_size or use gradient accumulation.
CPU Training: Possible but slow (~30–60 mins for 1,000 examples vs. ~5–10 mins on GPU).


Experiment Gradually:

Test changes one at a time (e.g., increase batch_size, then adjust lr).
Use a validation set (e.g., MS MARCO’s validation split) to check generalization.


Dataset Size:

The code uses 1,000 examples for speed. For better results, increase subset_size to 10,000 or use the full dataset (~82K examples).
More data may require more epochs (5–10) but improves answer quality.


Overfitting vs. Underfitting:

Overfitting: If training loss is low but answers are poor (test with inference code), reduce num_epochs or add weight decay.
Underfitting: If loss is high (>2 after 3 epochs), increase num_epochs or subset_size.




