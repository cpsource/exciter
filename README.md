# Exciter

This repository explores a custom neural network component called the **Exciter Layer** for pyTorch, inspired by recent research suggesting that **astrocytes** may play a role in **short-term memory** in biological systems.

## Overview

The Exciter Layer is a specialized hidden layer that:
- Includes a custom autograd function (`ExciterFunction`)
- Provides a mechanism for neuron-specific callbacks during backpropagation
- Can optionally bypass traditional learned weights and biases
- Is designed to support future extensions for memory-like behavior

The goal of this work is to investigate whether embedding a form of "pre-knowledge" or structured feedback into a hidden layer could improve training efficiency and model generalization.

## Files

- `exciter.py`: Initial version of the implementation (developed with Claude 4).
- `exciter_chat4.py`: Improved and debugged version with enhancements made using ChatGPT-4o.
- `ForwardProp.png`, `BackwardProp.png`: Visual diagrams showing the forward and backward propagation paths.

## Status

- ✅ Custom autograd function and callback mechanism implemented
- ✅ Optionally toggled masked or passthrough Exciter-to-Hidden connections
- ❌ Short-term memory functionality not yet implemented

## Future Directions

The next planned step is to explore how short-term memory dynamics could be added to the Exciter layer—potentially by incorporating time-dependent or stateful behavior directly into the autograd logic.

## Motivation

This project stems from the hypothesis that neural networks could benefit from biologically inspired memory mechanisms. By modeling astrocyte-like behavior in intermediate layers, the hope is to make networks:
- Faster to train
- More capable of handling temporal context
- More interpretable in how intermediate representations evolve

---

Feel free to use, modify, or extend this architecture for your own experiments. Contributions and ideas are welcome.



