# RNN-TensorFlow--Low-level

This notebook is a low-level implementation of a RNN using TensorFlow 2.0 with a custom training loop and which generates sequences from the trained network.

## Language Modeling & Recurrent Neural Networks

Language Modeling forms an important basis for most NLP applications such as tagging, parsing or machine translation. However, it can also be used on its own to generate “natural” language. The script is inspired by [this famous blog post by Andrej Karpathy "The Unreasonable Effectiveness of Recurrent Neural Networks"](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) for Character-Level Language Models. 

The RNN is trained to predict the next element of a sequence given the previous elements. That is, at each time step the RNN receives a character as input. From this input and its current state, it computes a new state and produces a probability distribution over the next character. Later, we can generate sequences by sampling single elements from the RNN’s output probability distribution and feeding them back into the network as input.

This script implements a RNN from the ground up, defining variables and using basic operations such as tf.matmul to define the computations at each time step and over a full input sequence.

The trained RNN is used to generate language by sampling from the language model. 

The dataset used is all the works of Shakespeare concatenated into a single (4.4MB) file and is borrowed from [here](https://cs.stanford.edu/people/karpathy/char-rnn/). 
