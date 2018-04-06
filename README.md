# ENEE633 Projet 1: Fashion-MNIST classifcation  
## Author Xuyi Ruan 11/18/17 - 12/10/17

![](https://github.com/ruanxuyi/ml_mnist_fash/blob/master/p2/MNIST_fash.png)

-	Wrote code in Python to compare classification models on Fashion-MNIST dataset on TensorFlow;
-	Started with SVM, a light-weight algorithm on 50,000/10,000 training and test set, applied dimension reduction 
with PCA and achieved reasonable test accuracy of 88.7%;
-	Experimented with more computational expensive ResNet liked deep learning model, on a EC2 GPU instance, 
which improved the accuracy to 93.81%, where state-of-art was 95%;

[Part1 report](https://github.com/ruanxuyi/ml_mnist_fash/blob/master/p1/README.md)  
[Part2 report](https://github.com/ruanxuyi/ml_mnist_fash/blob/master/p2/README.md)  

## Introduction

MNIST handwritten dataset is one of the classical data set to benchmark classification model in machine learning. As GPUs became more accessible, many of the neural network models could easily achieved $99.0\%$ or above accuracy on MNIST handwritten number dataset. Because of this, the MNIST fashion dataset emerged.  

Fashion-MNIST, a MNIST like dataset, has the same format (training set of 60,000 examples and a test set of 10,000 examples, input size $28\times28$, and $10$ output classes) as MNIST handwritten dataset but slightly more complicated image contents. The intension of this dataset is to make more distinguish performance for different types of classifers. We will explore Bayes, nearest neighbor, SVM(linear and non-linear) and various neural network models and compare their performance on Fashion-MNIST dataset in this project. 


## Fashion-MNIST classifcation Part 1

| Classifier Type   | Test Accur.  | Train time | Config.  |
| :------------- |:-------------:| -------------:| -----:|
| LDA | 81.50% | 0.16 mins | NA |
| SVM Linear w/PCA   | 85.37% | 9.94 mins | $c=1, \gamma=0.025$ |
| SVM Poly w/PCA | 88.59%      |   2.26 mins | $c=10, \gamma=0.025$ |
| SVM RBF w/ PCA | 88.77%      |   4.33 mins | $c=10, \gamma=0.025$ |
| LeNet/CNN	 | 91.54%     |    104.87 mins | detail above |
| **ResNet** | **93.81%**     |     2:50:26s | detail above |
| VGG | 91.67%*     |    1:22:23s* | detail above |

> SVM, CNN, LeNet ran on AWS EC2 CPU instance `t2.medium` 2CPUs + 4.0GB RAM  

> ResNet and VGG ran on AWS EC2 GPU spot instance with NVIDIA GRID K520 4GB RAM + 8CPUs + 16GB RAM 

> VGG spot instance reclaimed by amazon and did not finish all training (epoach [18/100]). Picked best test accuracy on among 18 epochs.

## Fashion-MNIST classifcation Part 2
### Experiment Results for Bayes classifier

| Classifier Type   | Test Accur.  | Train time | Config.  |
| :------------- |:-------------:| -------------:| :-----|
| GaussianNB   | 76.62% | 2.87 seconds | NA |
| GaussianNB + PCA | 76.68%    |   0.28 seconds | $pca=50$ |
| **GaussianNB + LDA** | **81.09%**    |   **0.07 seconds** | $NA$ |

### Experiment Results for kNN

| Classifier Type   | Test Accur.  | Train time | Config.  |
| :------------- |:-------------:| -------------:| :-----|
| kNN | 85.38%    |   399.38 seconds | $k = 10$ |
|**kNN + PCA** | **86.64%**   |   **13.42 seconds** | $pca=110, k = 10$ |
| kNN + LDA | 83.10%    |   0.80 seconds | $NA$ |


## Summary

| Classifier Type   | Test Accur.  | Train time | Config.  |
| :------------- |:-------------:| -------------:| -----:|
| **GaussianNB + LDA** | 81.09%    |   **0.07 seconds** | $NA$ |
|kNN + PCA | 86.64%   |   13.42 seconds | $pca=110, k = 10$ |
| LDA | 81.50% | 0.16 mins | NA |
| SVM Linear w/PCA   | 85.37% | 9.94 mins | $c=1, \gamma=0.025$ |
| SVM Poly w/PCA | 88.59%      |   2.26 mins | $c=10, \gamma=0.025$ |
| SVM RBF w/ PCA | 88.77%      |   4.33 mins | $c=10, \gamma=0.025$ |
| LeNet/CNN	 | 91.54%     |    104.87 mins | detail above |
| **ResNet** | **93.81%**     |     2:50:26s | detail above |
| VGG | 91.67%*     |    1:22:23s* | detail above |
