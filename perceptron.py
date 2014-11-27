The MIT License (MIT)

# Copyright (c) 2014 Jake Cowton

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from numpy.random import random_sample
from math import exp

class Perceptron(object):

    def __init__(self, no_of_inputs):
        """
        Initialises the weights with random values
        Sets the learning rate
        """
        self.w = random_sample(no_of_inputs + 1) # R, G, B + bias
        self.lr = 0.001
        self.bias = float(1)

    def weight_adjustment(self, inputs, error):
        """
        Adjusts the weights in self.w
        @param inputs a list of the input values used
        @param error the difference between desired and calculated
        """
        for x in range(len(inputs)):
            # Adjust the input weights
            self.w[x] = self.w[x] + (self.lr * inputs[x] * error)

        # Adjust the bias weight (the last weight in self.w)
        self.w[-1] = self.w[-1] + (self.lr * error)

    def result(self, inputs):
        """
        @param inputs one set of data
        @returns the the sum of inputs multiplied by their weights
        """
        value = 0
        for x in range(len(inputs)):
            # Add the value of the inputs
            value += inputs[x] * self.w[x]

        # Add the value of bias
        value += self.bias * self.w[-1]

        # Put value into the SIGMOID equation
        return float(1/(1+exp(-value)))

    def recall(self, inputs):
        res = self.result(inputs)
        if res > 0.5:return 'BLUE'
        elif res <= 0.5: return 'RED'
        else: return 'FAIL'