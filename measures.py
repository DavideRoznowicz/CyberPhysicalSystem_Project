import pandas as pd
import numpy as np
from cmath import inf
from moonlight import *
import pid_model
from Pendolum_Model import Pendolum



def perform_robustness(
        pid_model, 
        angle,
        mons, 
        ref, 
        initial_values, 
        noise, 
        max_exceed_angle=0.08, 
        num_times=400
    ):
    
    """
    Perform robustness analysis on given starting and target angle. Runs pid model and tracks
    relevant variables. Exploits Moonlight tracking and returns minmax robustness values.

    Input:
        - pid_model (obj): instance of class PID
        - angle (float): target angle in degrees
        - mons (list): moonlight monitors which track different signals
        - ref (np.array): target signal
        - initial_values (list): initial value of theta and omega, e.g. [theta0, omega0]
        - noise (np.array): noise added to theta estimates
        - max_exceed_angle (float): maximum admitted value that we are ok with if (in case of overshoot)
        - num_times (int): number of timesteps

    Output:
        - (tuple of list): contains robustness minmax values

    """
    

    theta_store, theta_store_with_noise, omega_store, ref_store, ts = pid_model.run(angle, num_times-1, num_times, initial_values, noise)

    
    # PHI 1
    diff = [np.abs(theta_store[i] - theta_store[i-1]) for i in range(1, len(ts))]
    diff.insert(0,0)
    diff = list(zip(diff))

    res_vect = np.array(mons[0].monitor(list(ts), diff))
    min_1, max_1 = res_vect[:,1].min(), res_vect[:,1].max()



    # PHI 2
    diff = np.abs(theta_store - ref_store)
    diff = list(zip(diff))

    res_vect = np.array(mons[1].monitor(list(ts), diff))
    min_2, max_2 = res_vect[:,1].min(), res_vect[:,1].max()



    # PHI 3
    theta_max = ref_store - max_exceed_angle
    diff = list(zip(theta_store, theta_max))

    res_vect = np.array(mons[2].monitor(list(ts), diff))
    min_3, max_3 = res_vect[:,1].min(), res_vect[:,1].max()


    return ([min_1, min_2, min_3], [max_1, max_2, max_3])



def perform_multiple_robustness(
        pid_model,
        mons, 
        initial_values, 
        noise, 
        angles=[40,30,20,10], 
        max_exceed_angle=0.08, 
        num_times=400
    ):
    
    """
    Perform robustness analysis on given starting and target angles (several). Runs pid model and tracks
    relevant variables. Exploits Moonlight tracking and returns a dataframe with main robustness
    results.

    Input:
        - pid_model (obj): instance of class PID
        - mons (list): moonlight monitors which track different signals
        - initial_values (list): initial value of theta and omega, e.g. [theta0, omega0]
        - noise (np.array): noise added to theta estimates
        - angles (list of floats): target angles in degrees
        - max_exceed_angle (float): maximum admitted value that we are ok with if (in case of overshoot)
        - num_times (int): number of timesteps

    Output:
        - df (pd.DataFrame): contains pandas dataframe with a summary of minmax robustness values
          for all of the initial starting angles

    """
    
    assert len(angles)==4
    
    
    nan = float("NAN")
    data = {
        '1': {'PHI_1': nan, 'PHI_2': nan, 'PHI_3': nan},
        '2': {'PHI_1': nan, 'PHI_2': nan, 'PHI_3': nan},
        '3': {'PHI_1': nan, 'PHI_2': nan, 'PHI_3': nan},
        '4': {'PHI_1': nan, 'PHI_2': nan, 'PHI_3': nan},
        '5': {'PHI_1': nan, 'PHI_2': nan, 'PHI_3': nan},
        '6': {'PHI_1': nan, 'PHI_2': nan, 'PHI_3': nan},
        '7': {'PHI_1': nan, 'PHI_2': nan, 'PHI_3': nan},
        '8': {'PHI_1': nan, 'PHI_2': nan, 'PHI_3': nan}
    }



    # Create the DataFrame with multilevel index
    df = pd.DataFrame(data)


    # Rename column names for each level
    df.columns = pd.MultiIndex.from_tuples([(f'Angle_{angles[0]}', 'Min Robustness'), 
                                            (f'Angle_{angles[0]}', 'Max Robustness'), 
                                            (f'Angle_{angles[1]}', 'Min Robustness'), 
                                            (f'Angle_{angles[1]}', 'Max Robustness'),
                                            (f'Angle_{angles[2]}', 'Min Robustness'), 
                                            (f'Angle_{angles[2]}', 'Max Robustness'), 
                                            (f'Angle_{angles[3]}', 'Min Robustness'), 
                                            (f'Angle_{angles[3]}', 'Max Robustness')])

    # for each angle, we run the pid model and compute robustness values with Moonlight
    for angle in angles:
        ref = np.zeros(num_times) + np.pi/180*angle
        
        min_col, max_col = perform_robustness(
            pid_model,
            angle,
            mons,
            ref, 
            initial_values, 
            noise, 
            max_exceed_angle=max_exceed_angle,
            num_times=num_times
        )

        df[(f'Angle_{angle}', 'Min Robustness')] = min_col
        df[(f'Angle_{angle}', 'Max Robustness')] = max_col

    return df
        




def rise_time(x, ref, time):
    """
    Point in time in which the output signal crosses the target.
    
    Input:
        - x (list): theta signal
        - ref (list): target signal
        - time (list): timesteps

    Output:
        - (float): time of crossing

    """
    i = 0
    if (x[0] > x[-1]):
        for x_val in x:
            if (x_val <= ref[i]):
                return round(time[i],4)
            i+=1
    else:
        for x_val in x:
            if (x_val >= ref[i]):
                return round(time[i],4)
            i+=1
            
    return float("NAN")

def overshoot(x, ref):
    """
    Difference between the max value of the system output and the target
    
    Input:
        - x (list): theta signal
        - ref (list): target signal

    Output:
        - (float): overshoot

    """
    
    # Assuming the system is going towards convergence
    if (x[0] > x[-1]):
        ind = np.where(x == np.min(x))
        oversh = round(-(x[ind][0]-ref[ind][0]), 2)
    else:
        ind = np.where(x == np.max(x))
        oversh = round(x[ind][0]-ref[ind][0], 2)
    
    if oversh > 0:
        return oversh
    
    return float("NAN")


def steady_state_error(x, ref):
    """
    Difference between steady state value of the output signal and value of the reference signal
    
    Input:
        - x (list): theta signal
        - ref (list): target signal

    Output:
        - (float): steady state error

    """
    return round(np.abs(x[-1] - ref[-1]), 4)


def settling_time(x, time, threshold=0.0005): # check, se c'è troppa oscillazione non funziona
    """
    Time at which the output reaches its steady state value
    
    Input:
        - x (list): theta signal
        - ref (list): target signal
        - threshold (float): value to determine settling time

    Output:
        - (float): settling time

    
    """
    i = 0
    while x[i] > x[-1] + threshold or x[i] < x[-1] - threshold:
        i+=1
    return round(time[i], 4)


def display_performances(angle, theta_store, ref_store, ts):
    """
    Returns a dataframe with
    
    Input:
        - angle (int): angle in degrees
        - theta_store (list): theta values
        - ref_store (list): target signal
        - ts (list): timesteps

    Output:
        - df (pd.DataFrame): contains main metrics of interest in produces signal

    
    """
    values = [
        overshoot(theta_store, ref_store),
        rise_time(theta_store, ref_store, ts),
        steady_state_error(theta_store, ref_store),
        settling_time(theta_store, ts)
    ]
    
    data = {
        f'Angle_{angle}': values
    }
    
    df = pd.DataFrame(data, index=['overshoot', 'rise time', 'steady state error', 'settling time'])

    return df