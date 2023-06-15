#--------------------------------------------------#
# This file contains the class describing the
# PID control model.
# 
# It exploits the Pendolum model and its
# differential equations to build a control for it
# and adapt behaviour of the force u to achieve a
# certain goal
#--------------------------------------------------#


#----------------------# import standard python packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
#----------------------#


#----------------------# import custom python functions
from Pendolum_Model import Pendolum
#----------------------#



class PID():
    """
    PID control class building a proper control u which reacts swiftly to changes in underlying
    model behaviour, adapting the output to achieve our intended objective
    """
    def __init__(self, model, Kp, Ki, Kd): # Ki = Kc/taui -Kc * taud
        """
        Initialization of PID model with K params for control.

        Input:
            - model (list): Pendolum Model in this context 
            - Kp (float): error component constant
            - Ki (float): integral component constant
            - Kd (float): derivative component constant

        Output:
            -


        """
        self.model = model
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd


    def run(self, angle, tf, nsteps, initial_values, noise, sin=False):
        """
        Running PID model

        Input:
            - angle (int): starting angle in degrees
            - tf (int): final time for simulation 
            - nsteps (int): number of timesteps 
            - initial_values (list): initial values like this: [theta0, omega0]
            - noise (list): noise injected into theta estimates
            - sin (bool): whether to use a default sinusoidal reference signal instead
              of the standard one
        Output:
            - theta_store (np.array): array of stored theta values during simulation
            - theta_store_with_noise (np.array): array of stored theta values injected with noise
            - omega_store (np.array): array of stored omega values during simulation
            - ref_store (np.array): array of values of reference signal
            - ts (list): timesteps 
            
        """

        # time
        ts = np.linspace(0,tf,nsteps)
        delta_t = tf/(nsteps)   # length of each timestep

        # values of interest for PID
        sum_int, error, u = 0.0, 0, 0
        
        # store history
        ref = np.zeros(nsteps) + np.pi/180*angle
        if sin:
            ref = [np.sin(i/20)/10 + 0.2 for i in range(nsteps)]
        ref_store = np.zeros(nsteps)
        omega_store = np.zeros(nsteps)
        theta_store = np.zeros(nsteps)
        theta_store_with_noise = np.zeros(nsteps)


        numer = 0
        for i in range(nsteps):
            
            ref_store[i] = ref[i]
            omega_store[i] = initial_values[1] 
            theta_store[i] = initial_values[0]
            initial_values[0] = initial_values[0] + noise[i]
            theta_store_with_noise[i] = initial_values[0]
            
            error = ref[i] - initial_values[0]
            sum_int += error*delta_t
            
            if i > 0:
                numer = (initial_values[0]-theta_store[i-1]) # with observed for the derivative kick

            u = self.Kp*error + self.Ki * sum_int + self.Kd * numer / delta_t

            theta_omega = odeint(self.model.differential_theta, initial_values,[0,delta_t],args=(u,))
            initial_values = theta_omega[-1]

        return theta_store, theta_store_with_noise, omega_store, ref_store, ts

    
    def plot(self, angle, theta_store, theta_store_with_noise, omega_store, ref_store, ts):
        """
        Plotting the functioning theta values (or signals in case of noise)

        Input:
            - angle (int): starting angle in degrees
            - theta_store (np.array): array of stored theta values during simulation
            - theta_store_with_noise (np.array): array of stored theta values injected with noise
            - omega_store (np.array): array of stored omega values during simulation
            - ref_store (np.array): array of values of reference signal
            - ts (list): timesteps 

        Output:
            - None


        """
        # plot results
        plt.rcParams["figure.figsize"] = (10,6)
        

        plt.plot(ts,theta_store,'g-',linewidth=2)
        plt.plot(ts, ref_store,'k--',linewidth=1)
        
        if sum(abs(theta_store_with_noise - theta_store)) > 0.0001:
            plt.plot(ts, theta_store_with_noise, 'r-', linewidth=2)
            plt.legend(["theta", "reference", "theta perturbated"])
        
        plt.title(f"Angle_{angle}")
        plt.xlabel('Time')
        plt.ylabel('Theta (rad)')
        plt.show()