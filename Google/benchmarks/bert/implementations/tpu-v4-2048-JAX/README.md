# Lingvo-JAX large language model

## Benchmark Information

Language large model description

## Software

[JAX](https://github.com/google/jax) with [Lingvo](https://github.com/tensorflow/lingvo)

### Publication/Attribution

[Lingvo: a Modular and Scalable Framework for Sequence-to-Sequence Modeling]
(https://arxiv.org/pdf/1902.08295.pdf)

## Hardware

TPU v4.

## Model

Different from the reference BERT implementation, this submission uses a 66
layer encoder only model, with model dimension 12288,
hidden dimension 98304 and 128 attention heads for each layer.
Detailed model configuration can be found in file

```
lingvo/jax/tasks/lm/params/bert.py
```

class 

```
BertSpmdL66H12kBiggerBatch8x8x16
```

The total nuber of parameters is 199,717,148,096.

We also selected different initial checkpoint and stopping criterion from the
closed division reference implementation, and will have a seperated document
explaining them.

The optimizer used is Adafactor instead of LAMB in the closed division.

## Dataset

The dataset is the same as the one used in the closed division.

## Research Submissions

JAX research submissions were run using Google internal infrastructure. Contact
Peter Mattson (petermattson@google.com) for more details.
Ref: CL/404060414

## Instructions to run

See 

```
run_and_time.sh
```


