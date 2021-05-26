import numpy as np
import sys

def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block)
                                                    , round(progress*100,2), status)
    sys.stdout.write(text)
    sys.stdout.flush()

class NumericalLangevin():
    def __init__(self, fcn_force=None, fcn_diff=None, population=None):
        self.fcn_force = fcn_force
        self.fcn_diff = fcn_diff
        self.population = population

    def set_params(self, fcn_force=None, fcn_diff=None, population=None):
        if fcn_force is not None: self.fcn_force = fcn_force
        if fcn_diff is not None: self.fcn_diff = fcn_diff
        if population is not None: self.population = population

    def wiener_process(self, timeStep):
        dW = np.sqrt(timeStep)*np.random.normal(0.0, 1.0, len(self.population))
        return dW

    def euler_scheme(self, nbrSteps, timeStep):
        trajectory  = np.zeros( (len(self.population), nbrSteps) )
        trajectory[:,0] = np.copy(self.population)
        checkZero = 0
        for i in np.arange(1, nbrSteps):
            nextTraj = trajectory[:,i-1] + (self.wiener_process(timeStep)
                                * self.fcn_diff(trajectory[:,i-1])
                                + self.fcn_force(trajectory[:,i-1])*timeStep )
            # count number of times a zero appears
            if np.any(nextTraj<0): checkZero += 1
            # set negatives to zero
            nextTraj[nextTraj<0.0] = 0.0; trajectory[:,i] = nextTraj
            update_progress(i/nbrSteps)
        print("\nNumber of zeros : " + str(checkZero) )

        return trajectory
