# A Neural Network to calculate if an RGB value is more red or blue
from perceptron import Perceptron

BLUE = 1
RED = 0
# Lowest MSE
LMSE = 0.001

def normalise(data):
    """
    MUST BE CUSTOMISED PER PROJECT
    Turn data into values between 0 and 1
    @param data list of lists of input data and output e.g.
        [
            [[0,0,255], 1],
            ...
        ]
    @returns Normalised training data
    """
    temp_list = []
    for entry in data:
        entry_list = []
        for value in entry[0]:
            # Normalise the data. 1/255 ~ 0.003921568
            entry_list.append(float(value*0.003921568))
        temp_list.append([entry_list, entry[1]])
    return temp_list

def main(data):
    # Normalise the data
    training_data = normalise(data)
    # Create the perceptron
    p = Perceptron(len(data[0][0]))

    # Number of full iterations
    epochs = 0
    # Instantiate mse for the loop
    mse =999

    while (abs(mse-LMSE) > 0.002):

        # Epoch cumulative error
        error = 0

        # For each set in the training_data
        for value in training_data:
            # Calculate the result
            output = p.result(value[0])
            # Calculate the error
            iter_error = value[1] - output
            # Add the error to the epoch error
            error += iter_error

            # Adjust the weights based on inputs and the error
            p.weight_adjustment(value[0], iter_error)

        # Calculate the MSE - epoch error / number of sets
        mse = float(error/len(training_data))

        # Print the MSE for each epoch
        print "The MSE of %d epochs is %.10f" % (epochs, mse)

        # Every 100 epochs show the weight values
        if epochs % 100 == 0:
            print "0: %.10f - 1: %.10f - 2: %.10f - 3: %.10f" % (p.w[0], p.w[1], p.w[2], p.w[3])

        # Increment the epoch number
        epochs += 1

    return p
