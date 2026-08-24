# This is a constants file for the gas network simulation. 
# It contains all physical constants used in the formulation of the equation system.
# If this solver is used for simulation of other types of networks, adjust the constants accordingly.

class GasConstants:
    R = 8.31446261815324    # J/(mol*K) - universal gas constant
    M = 0.002016            # kg/mol - molar mass of hydrogen (H2)
    T = 288.15              # K - standard temperature
    P = 101325              # Pa - standard pressure
    g = 9.81                # m/s^2 - acceleration due to gravity
    PA_PER_BAR = 100_000    # Pa/bar - pressure in the model is expressed in bar, not Pa