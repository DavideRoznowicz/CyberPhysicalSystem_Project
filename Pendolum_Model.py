#--------------------------------------------------#
# This file contains the class describing the
# Pendolum behaviour via its differential equations.
# 
# The usage of this class comes handy when
# solving the related differential equations within
# the PID control model.
#--------------------------------------------------#


#----------------------# import standard python packages
import numpy as np
#----------------------#





class Pendolum():
    """
    Pendolum class describing the behaviour of a Pendolum made of an inextensible wire
    connected to a small ball at the extremity. The differential equations model several
    forces the pendolum is subject to, in particular the ball weight, friction forces
    and u control.
    """

    def __init__(self, frct, m, g, r):
        """
        Initialization of the parameters characterizing the pendolum.
        
        Input:
            - frct (float): friction parameter
            - m (float): mass of the ball attached at the extremity of the wire
            - g (float): value gravitational acceleration
            - r (float): length of the inextensible wire
        
        Output:
            - (None)
        
        """
        
        self.frct = frct
        self.m = m
        self.g = g
        self.r = r

    def differential_theta(self, theta, t, u):
        """
        Computation of the derivative of theta and omega in time. A list containing the
        two values is eventually returned.
        
        Input:
            - self (obj): model object
            - theta (list): list containing the value of theta and its
                            derivative omega, i.e. [theta, omega]
            - t (float): time value
            - u (float): value of our own control
        
        Output:
            - [dtheta_dt, domega_dt] (list): list containing the derivative of theta and
                                             the derivative of omega on time
        
        """
        
        # derivative of theta in time (== omega)
        dtheta_dt = theta[1]
        
        # derivative of omega in time
        domega_dt = ((u/(self.m*self.r)) - (self.frct*theta[1]/self.m) - (self.g*np.sin(theta[0]))/self.r)

        return [dtheta_dt, domega_dt]
