<div align="center">

  <h1>QuantumML</h1>
  
  <p>
    Welcome to the Quantum Machine Learning Toy Example Repository
  </p>
  
  
<!-- Badges -->
<p>
  <a href="https://github.com/jjprietotorres/QuantumML/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/jjprietotorres/QuantumML" alt="contributors" />
  </a>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/jjprietotorres/QuantumML" alt="last update" />
  </a>
  <a href="https://github.com/jjprietotorres/QuantumML/network/members">
    <img src="https://img.shields.io/github/forks/jjprietotorres/QuantumML" alt="forks" />
  </a>
  <a href="https://github.com/jjprietotorres/QuantumML/stargazers">
    <img src="https://img.shields.io/github/stars/jjprietotorres/QuantumML" alt="stars" />
  </a>
  <a href="https://github.com/jjprietotorres/QuantumML/issues/">
    <img src="https://img.shields.io/github/issues/jjprietotorres/QuantumML" alt="open issues" />
  </a>
  <a href="https://github.com/jjprietotorres/QuantumML/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/jjprietotorres/QuantumML.svg" alt="license" />
  </a>
</p>
   
<h4>
    <!--<a href="https://github.com/jjprietotorres/QuantumML/">View Demo</a>
  <span> · </span>-->
    <a href="https://github.com/jjprietotorres/QuantumML">Documentation</a>
  <span> · </span>
    <a href="https://github.com/jjprietotorres/QuantumML/issues/">Report Bug</a>
  <span> · </span>
    <a href="https://github.com/jjprietotorres/QuantumML/issues/">Request Feature</a>
  </h4>
</div>

<br />

<!-- Table of Contents -->
# :notebook_with_decorative_cover: Table of Contents

- [About the Project](#star2-about-the-project)
- [Getting Started](#toolbox-getting-started)
  * [Prerequisites](#bangbang-prerequisites)
  * [Run Locally](#running-run-locally)
- [Usage](#eyes-usage)
- [License](#warning-license)
- [Contact](#handshake-contact)
- [Acknowledgements](#gem-acknowledgements)

<!-- About the Project -->
## :star2: About the Project

This repository contains a toy example of using Quantum Machine Learning (QML) to improve the performance of kernel calculations in a Support Vector Machine (SVM) classifier. The code is written in Python and utilizes the IBM Qiskit library to implement the QML algorithms.

In this example, we compare the performance of a classical SVM binary classifier using kernel optimization to the performance of a QML-enhanced SVM classifier on non-linear data. The results demonstrate that the use of QML can significantly improve the accuracy of the classification.

This example is meant to be a simple introduction to the possibilities of QML and how it can be applied to real-world problems. We hope you find it informative and useful.

In the context of SVM classifiers, a kernel is a function that is used to transform the input data into a higher dimensional space, where it becomes linearly separable. This allows the classifier to perform better on non-linear data by mapping it to a higher dimensional space in which a linear decision boundary can be found.

The optimization of the kernel function is crucial for the performance of the SVM classifier. The optimization process is computationally expensive and can take a long time for large datasets.

Quantum Machine Learning (QML) offers a way to improve the performance of kernel optimization by leveraging the principles of quantum computing. The basic idea is to use quantum algorithms to perform the optimization more efficiently and with a higher accuracy.

In this toy example, we use the IBM Qiskit library to implement a quantum-enhanced kernel optimization algorithm. The results show that the use of QML can significantly improve the accuracy of the SVM classifier on non-linear data. This highlights the potential of QML in providing faster and more accurate solutions to real-world problems.

<!-- TechStack -->
### :space_invader: Tech / Model Stack

#### What is a kernel function?

Kernel methods are a collection of pattern analysis algorithms that use kernel functions to operate in a high-dimensional feature space. The best-known application of kernel methods is in **Support Vector Machines (SVMs)**, supervised learning algorithms commonly used for classification tasks. The main goal of SVMs is to find decision boundaries to separate a given set of data points into classes. When these data spaces are not linearly separable, SVMs can benefit from the use of kernels to find these boundaries.
    
Formally, decision boundaries are hyperplanes in a high dimensional space. The kernel function implicitly maps input data into this higher dimensional space, where it can be easier to solve the initial problem. In other words, kernels may allow data distributions that were originally non-linearly separable to become a linearly separable problem. This is an effect known as the kernel trick.
    
There are use-cases for kernel-based unsupervised algorithms too, for example, in the context of clustering. **Spectral Clustering** is a technique where data points are treated as nodes of a graph, and the clustering task is viewed as a graph partitioning problem where nodes are mapped to a space where they can be easily segregated to form clusters.
    
  ### Kernel Functions:
    
  Mathematically, kernel functions follow:
    
  $k(\\vec{x}_i, \\vec{x}_j) = \\langle f(\\vec{x}_i), f(\\vec{x}_j) \\rangle$,
    
  where,
  * $k$ is the kernel function,
  * $\\vec{x}_i, \\vec{x}_j$ are $n$ dimensional inputs,
  * $f$ is a map from $n$-dimension to $m$-dimension space and
  * $\\langle a,b \\rangle$ denotes the inner product,
    
    
  When considering finite data, a kernel function can be represented as a matrix:
  
  $K_{ij} = k(\\vec{x}_i,\\vec{x}_j)$.
  
  ### Quantum Kernels
  
  The main idea behind quantum kernel machine learning is to leverage quantum feature maps to perform the kernel trick. In this case, the quantum kernel is created by mapping a classical feature vector $\\vec{x}$ to a Hilbert space using a quantum feature map $\\phi(\\vec{x})$. Mathematically:
  
  $K_{ij} = \\left| \\langle \\phi(\\vec{x}_i)| \\phi(\\vec{x}_j) \\rangle \\right|^{2}$
  
  where
  * $K_{ij}$ is the kernel matrix,
  * $\\vec{x}_i, \\vec{x}_j$ are $n$ dimensional inputs,
  * $\\phi(\\vec{x})$ is the quantum feature map,
  * $\\left| \\langle a|b \\rangle \\right|^{2}$ denotes the overlap of two quantum states $a$ and $b$
    
  Quantum kernels can be plugged into common classical kernel learning algorithms such as SVMs or clustering algorithms, as you will see in the examples below. They can also be leveraged in new quantum kernel methods like `QSVC`

#### What is IBM qiskit?

IBM Qiskit is an open-source quantum computing framework for creating and running quantum algorithms. It is developed by IBM and is designed to be used by both researchers and developers to build and run quantum programs.

Qiskit is made up of four main components:

- Terra: The foundation of Qiskit, Terra provides the basic building blocks for creating quantum circuits and executing them on quantum devices or simulators.

- Aer: A high-performance simulator that allows for the execution of quantum circuits on classical computers.

- Ignis: A toolset for analyzing and mitigating errors in quantum circuits.

- Aqua: A library of algorithms for various domains such as optimization, chemistry and machine learning.

Qiskit is widely adopted by the quantum computing community and it is considered the one of the most widely used open-source libraries for quantum computing. It supports a wide range of quantum computing platforms and simulators, including IBM's own cloud-based quantum computing platform, IBM Q.

This allows researchers and developers to test their algorithms on real quantum devices and simulators, and to access the latest advances in quantum computing.


#### Model Details

##### Support Vector Machine (SVM)

A Support Vector Machine (SVM) is a type of supervised learning algorithm that can be used for classification or regression tasks. It works by finding the best boundary (called a "hyperplane") that separates the different classes in the data. The best boundary is the one that maximizes the margin, which is the distance between the boundary and the closest points of each class (called "support vectors").

SVMs are particularly useful for datasets that are not easily separated by a linear boundary. In these cases, the algorithm can transform the data into a higher-dimensional space using a function called a kernel, where a linear boundary can be found.

In this toy example, we use the SVM algorithm for a binary classification task. A binary classification task is a task in which the algorithm must classify the data into one of two classes.

##### Binary Classifier SVC

One of the most common types of SVM classifiers is the Support Vector Classification (SVC) algorithm. SVC is used for binary classification tasks, where the goal is to separate the data into two classes.

The SVC algorithm finds the best boundary by maximizing the margin between the two classes. This boundary is called the "decision boundary" and separates the data into the two classes.

The basic idea behind SVC is to find the best boundary (or hyperplane) that separates the data into the two classes, while maximizing the margin.

A simple example of a binary classifier SVC is a classifier that separates apples and oranges. The SVC algorithm would find the best boundary that separates the apples and oranges based on their features, such as color, shape, and size.

The SVC algorithm can be implemented in Python using the sklearn.svm.SVC class. The basic usage is to create an instance of the class, set the kernel, fit the model using the training data and then use the predict method to predict the class of the test data.

```python
    from sklearn.svm import SVC

    # Create a SVC classifier using a radial basis function (RBF) kernel
    clf = SVC(kernel='rbf')

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    # Use the classifier to predict the class of the test data
    y_pred = clf.predict(X_test)

```
This is a very simple example of how the SVC algorithm can be used for binary classification tasks. The actual implementation may require more advanced techniques such as cross validation and fine-tuning the parameters.

In this toy example we are going to use a non-linear kernel function for the dataset, and we will compare the performance of the classical SVM and the QML-enhanced SVM on the same non-linear data.

#### Training Data
Using the following code, we will create a simple dataset that has the form of an XOR gate using the logical_xor function from NumPy, to generate a toy non-linear dataset of 250 samples with 2 features. The dataset is generated by creating random values from a normal distribution using the np.random.randn function. This function creates an array of random numbers with a mean of 0 and a standard deviation of 1.

```python
    np.random.seed(SEED)
    X = np.random.randn(250, 2)
    y = np.logical_xor(X[:, 0] > 0,
                        X[:, 1] > 0)
    y = np.where(y, 1, -1)
```

The np.logical_xor function is used to create the labels for the samples. It compares the value of the first feature (X[:, 0]) with 0 and the value of the second feature (X[:, 1]) with 0, and returns a boolean array indicating where the values are greater than 0.

This generates a dataset where the samples are labeled as 1 if either the first feature or the second feature is greater than 0, and as -1 otherwise. The np.where function is used to convert the boolean array to an array of integers (1 or -1)

It creates a dataset where the samples are labeled as 1 if either the first feature or the second feature is greater than 0, and as -1 otherwise. This dataset is also non-linear, and it would be useful to test classifiers on it as well.

The XOR (Exclusive OR) operation is a non-linear operation because it cannot be represented by a linear combination of its inputs. A linear combination is a sum of inputs multiplied by some coefficients, such as f(x) = ax + by + c . A linear function can be represented by a straight line or a plane in a two-dimensional or three-dimensional space, respectively.

The XOR function, generates a result that is dependent on the specific combination of its inputs. The output is 1 when one input is 1 and the other input is 0, and 0 otherwise. This means that for any given input, the output will only be 1 for a specific set of input values. This set of input values cannot be represented by a straight line or a plane, but instead, it forms a non-linear decision boundary.

A simple example of this can be visualized by a 2D input space with two features (x1, x2), where x1 and x2 can take values of 0 or 1. The XOR operation returns 1 if one of the inputs is 1 and the other is 0, and 0 otherwise. Plotting the input space, the samples that belong to class 1 form a region that follows an "X" shape, this shape cannot be separated by a line but it can be separated by more complex decision boundaries.

Therefore, the XOR operation makes it useful for testing classifiers that can handle non-linear decision boundaries.

<!-- Getting Started -->
## 	:toolbox: Getting Started

<!-- Prerequisites -->
### :bangbang: Prerequisites

To get started, you will need to have Python (3.8) installed on your virtual enviroment. You will also need to install the IBM Qiskit library and other dependencies by running:

```bash
  pip install -r requirements.txt
```

<!-- Run Locally -->
### :running: Run Locally

1. Clone the project

```bash
  git clone https://github.com/jjprietotorres/QuantumML.git
```

2. Go to the project directory

```bash
cd src
```

3. Run the toy example (classical computing)

```bash
python classicTrain.py
```

4. Run the toy example (quantum computing)

```bash
python quantumTrain.py
```

<!-- Usage -->
## :eyes: Usage

Comparing the results of the classical SVM and the QML-enhanced SVM on the same non-linear data:

<!-- License -->
## :warning: License

Distributed under the MIT License. See [LICENSE](https://github.com/jjprietotorres/QuantumML/blob/master/LICENSE) for more information.

<!-- Contact -->
## :handshake: Contact

Juan José Prieto Torres - [@jjprietotorres](https://twitter.com/jjprietotorres) - jjprietotorres@gmail.com

Project Link: [https://github.com/jjprietotorres/QuantumML](https://github.com/jjprietotorres/QuantumML)

<!-- Acknowledgments -->
## :gem: Acknowledgements

Use this section to mention useful resources and libraries that you have used in your projects.

 - [Readme Template](https://github.com/Louis3797/awesome-readme-template)
 - [Qiskit doc.](https://qiskit.org/documentation/)
 - [Quantum kernel ML](https://github.com/Qiskit/qiskit-machine-learning/blob/main/docs/tutorials/03_quantum_kernel.ipynb)
 - [Github Repo](https://github.com/maximer-v/quantum-machine-learning)