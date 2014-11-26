# Single Layer Perceptron

SLPs are are neural networks that consist of only one neuron, the perceptron. It can take in an unlimited number of inputs and separate them linearly.

Here is a small bit of code from an assignment I'm working on that demonstrates how a single layer perceptron can be written to determine whether a set of RGB values are RED or BLUE.

Of course the G could just be ignored, but this code is just to show how a SLP can be used to get rid of noisy data and find the correct answer

## Training Data
Here's some example training data
```python
data = [#((R,G,B), CLASSIFICATION)
        [[0,0,255], BLUE],
        [[0, 0, 255], BLUE],
        [[0, 0, 192], BLUE],
        [[243, 80, 59], RED],
        [[255, 0, 77], RED],
        [[77, 93, 190], BLUE],
        [[255, 98, 89], RED],
        [[208, 0, 49], RED],
        [[67, 15, 210], BLUE],
        [[82, 117, 174], BLUE],
        [[168, 42, 89], RED],
        [[248, 80, 68], RED],
        [[128, 80, 255], BLUE],
        [[228, 105, 116], RED]
    ]
```

## Normalising Data

In this project my `normalise()` function takes in each of the input values and turns them into a value between 0 and 1. To do that I multiply each of the values by 0.003921568 because 1 / 255 ~ 0.003921568.

To modify the function for your own use, change out 0.003921568 for 1/(max input value).

## Increasing number of inputs
Theoretically, this can be done by passing the desired number of inputs into `Perceptron()` when you create it (I haven't tested this yet).