# XAI
Explainable Artificial Intelligence

## Guided backpropagation [![arXiv](https://img.shields.io/badge/arXiv-2101.02127v2-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/1412.6806)
STRIVING FOR SIMPLICITY: THE ALL CONVOLUTIONAL NET
>
```python
@tf.custom_gradient
def _GuidedReluGrad(x):
    def gradient(dy):
        return tf.cast(x>0, "float32") * tf.nn.relu(dy)
    return tf.nn.relu(x), gradient
```


## Integrated gradients [![arXiv](https://img.shields.io/badge/arXiv-2101.02127v2-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/1703.01365)
Axiomatic Attribution for Deep Networks
>


## Pettern Net [![arXiv](https://img.shields.io/badge/arXiv-2101.02127v2-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/1703.06339)
PatternNet: Visual Pattern Mining with Deep Neural Network
>

## Class-agnostic Activation Map (C<sup>2</sup>AM) [![arXiv](https://img.shields.io/badge/arXiv-2101.02127v2-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2203.13505) 
Contrastive learning of Class-agnostic Activation Map for Weakly Supervised Object Localization and Semantic Segmentation
> 

![image](https://user-images.githubusercontent.com/19265337/174466861-1542ca81-4578-4c9c-860e-4bea55b1c494.png)

