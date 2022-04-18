#==============================================================================
"""
Created on Tue May 15 2018
@author: Francisco J. Castellanos
"""
#==============================================================================


def assert_error():
    assert(False)

def assert_msg(msg):
    print ("Message error: " + str(msg))
    assert(False)

def assert_not_implemented():
    raise NotImplementedError("Please Implement this method")







#==============================================================================
# Code Tests...
#==============================================================================

if __name__ == "__main__":
    
    print("Main")