
# Simplified ECO Neural Network with binary step activation function #

Final performance on MNIST dataset:

* 250 training samples
* One hidden layer with 1000 neurons (almost same performance with two layers, each 1000 neurons)
* Ensemble averaging for each output value
* 1 minute training time
* 66% performance

Reference on ECO: [ECO Overview](https://blog.morphe.by/error-curvature-optimization-for-neural-networks/)

[![No Maintenance Intended](http://unmaintained.tech/badge.svg)](http://unmaintained.tech/)

## Principle ##

This network follows the same basic principle of ECO: instead of reducing total error value, it attempts
to increase the steepness of the error curve with regard to inputs.

For binary step, that means adjusting weights and biases in such a manner, that changes to
the input value will be more likely to change the output.

This network has a remarkable property of not being tied to the total error value, and each adjustment
is made strictly between a pair of neurons.

## Building ##

1. Download
2. Run `swift package generate-xcodeproj`
3. Open `Seconn.xcworkspace`
4. Adjust macOS Deployment target to macOS 10.12 (probably should fix this)
5. If desired, turn on Swift Optimizations for debug build (faster playground execution)
6. Build project
7. Open and use SeconnTest playground

## Configuring ##

Network configuration is set in `Test.swift`, in `Test.init`.
