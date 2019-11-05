#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    import numpy
    thresh = numpy.percentile((predictions - net_worths), 90.0)
    for i in range(0, len(ages)):
        age = ages[i]
        net_worth = net_worths[i]
        error = predictions[i] - net_worth
        if error < thresh:
            cleaned_data.append((age, net_worth, error))
        else:
            continue

    return cleaned_data

