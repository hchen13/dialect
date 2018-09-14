# Chinese Dialect Transformation using Deep Learning

## Abstract

Dialects in different regions of China are so distinct that they can cause considerable communication misunderstandings. To this end, this project attempts to transform the soundwaves of speeches spoken in Chengdu dialect into mandarin Chinese while preserving 
the original speed and intonation. The methodology adopted is a modified version of **wavenet** which is used in a variety range of applications, such as sound-to-text and music generation. 

中国方言众多, 且多大相径庭, 不同方言之间交流极为困难. 所以该项目尝试将成都话的原始音频数据转化为普通话, 并且保留其原始语速和语调. 实现方法基于应用广泛的**wavenet**, 该项目在此基础上做了部分调整.

## Neural network architecture

The basis of the neural network architecture used is wavenet, with a few modifications to better suit this particular problem. 

项目所使用的神经网络架构是在经过调整, 更适于这一特定问题的wavenet. 

In the original wavenet architecture, the dilated convolutions are applied on inputs with a kernel size of 2, and it maintains its causality by designing the architecture asymmetrical, in other words, a futue value is calculated and determined solely from a current value and a previous value from the last layer. The modified architecture removes the causality property simply by incrementing the kernel size to 3, having each current value be determined not only by a previous value and the current value, but also a value from the future symmetrical in position to the previous value used. This modification causes the architecture forms a symmetrical hierachy. The reason for this design is based on the fact that the pronunciation of a character spoken in mandarin Chinese is not only determined by the characters already spoken, but also the subsequent ones. 

原始的wavenet结构使用了核尺寸为2的扩展型卷积, 并且每个新的节点值是通过在上一层网络中的一个当前节点和一个前置节点计算得出, 该设计使神经网络形成了非对称的结构, 保持了神经网络的因果性. 而调整后的模型打破了因果性, 使用核尺寸为3的扩展性卷积作为计算方法, 且每个节点是通过前置, 当前, 后置节点的值计算得出, 该设计形成了对称的结构. 如此设计的原因是基于如下的事实: 在一句话中某个字的发音并不仅仅取决于已经说过的字, 还取决于即将说的内容. 

## Current status

Due to the lack of training data, the current model cannot generalize to arbitrary speakers other than the ones who recorded the training data, or phrases that have not appeared. 