import unit_tests.problem_unittests as tests
import numpy as np

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    # TODO: Implement Function
    #a = 0.0
    #b = 1.0
    #mi = 0.0
    ma = 255.0
    #return a + (((x - mi)*(b - a))/(ma - mi))
    return x/ma


"""
Test
"""
tests.test_normalize(normalize)


#encoder = LabelBinarizer()
#once = False
def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    #global once
    # TODO: Implement Function
    #if (once == False):
        #encoder.fit(x)
        #once = True

    #one_hor_labels = encoder.transform(x)
    #num_labels = np.max(x) + 1
    return np.eye(10)[x]


"""
Test
"""
tests.test_one_hot_encode(one_hot_encode)