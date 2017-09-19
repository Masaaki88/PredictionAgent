from numpy import ndarray
import numpy as np
import datetime
import os


def isNotFalse(arg):
    '''
    returns True except if arg==None or arg==False
    -> simplifies handling of ndarrays
    '''
    if isinstance(arg, ndarray):
        return True
    if arg:
        return True
    else:
        if isinstance(arg, bool) or arg==None:
            return False
        else:
            return True


def check_that_1d_float_array(arg, description):
    '''
    checks the following properties of arg:
        - arg is ndarray
        - arg is 1-dimensional
        - arg is float array
        - arg is not empty
    description for assertion output
    '''
    assert isinstance(arg, ndarray),\
        "\n{} ({}) has wrong type ({})!"\
        "\nType must be 'ndarray'!".format(description, arg, type(arg))
    assert arg.ndim == 1,\
        "\n{} ({}) has wrong dimension ({})!\
         \nDimension must be 1!".format(description, arg, arg.ndim)
    assert arg.dtype == float,\
        "\n{} ({}) contains wrong data type ({})!\
         \nData type must be 'float'!".format(description, arg, arg.dtype)
    assert len(arg) > 0,\
        "\n{} is empty!".format(description)



def check_that_contains_probabilities(arg, description):
    '''
    checks that arg contains probabilities, i.e. elements:
        - are finite,
        - are non-negative,
        - are smaller than or equal to 1.
    description for assertion output
    '''
    assert np.isfinite(arg).all(),\
        "\n{} contains invalid elements: "\
        "{}!".format(description, arg)
    assert (arg >= 0.0).all(),\
        "\n{} contains negative elements: "\
        "{}!".format(description, arg)
    assert (arg <= 1.0).all(),\
        "\n{} contains elements greater than 1: "\
        "{}!".format(description, arg)



def get_outputpath():
    '''
    create and return time-labelled folder for output

    outputpath is path to current output folder
    time_string is string of simulation start time
    '''
    cwd = os.getcwd()
 #   if cwd[-6:] != 'model2':
 #       cwd += '/model2'
    dt_now = datetime.datetime.now()
    today = '{}-{}-{}'.format(dt_now.year, dt_now.month, dt_now.day)
    t_0 = '{}:{}:{}'.format(dt_now.hour, dt_now.minute, dt_now.second)
    outputpath_short = cwd+'/output/'+today
                                        # date yields super folder
    outputpath = cwd+'/output/'+today+'/'+t_0+'/'
                                        # date + subfolder yields working folder
    for path_string in [outputpath_short, outputpath]:
        try:
            os.mkdir(path_string)
        except:
           pass
        
    time_string = '{} {}'.format(today, t_0)

    return outputpath, time_string
