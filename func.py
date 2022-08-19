import pandas as pd
from aircraftLib import JetAircraft
from UtilityFunctions import *
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import os
from matplotlib.backends.backend_pdf import PdfPages

constants = {}
constants["kappa"] = 1.4
constants["R"] = 287.05287
constants["g0"] = 9.8065
constants["lb_to_kg"] = 0.45359
constants["hr_to_sec"] = 1/3600
constants["mu"] = 1 / 3.5
constants["a0"] = 340.294
constants["beta"] = -0.0065
constants["p0"] = 101325
constants["rho0"] = 1.225
constants["T0"] = 288.15

kappa = constants["kappa"]
mu = constants["mu"]
a0 = constants["a0"]
beta = constants["beta"]
g0 = constants["g0"]
R = constants["R"]
T0 = constants["T0"]
p0 = constants["p0"]
rho0 = constants["rho0"]

datapath = "C:/Users/Recep/Desktop/Boeing Proje/Data"
files = [f for f in listdir(datapath) if isfile(join(datapath, f))]

ac = JetAircraft('B773ERGE115B')
for file in files:

    filepath = os.path.join("Data", file)
    data = pd.read_csv(filepath, low_memory=False)

    cl_rate = [0] * len(data.Time)
    cl_rate_real = [0] * len(data.Time)
    cr_rate = [0] * len(data.Time)
    d_rate = [0] * len(data.Time)
    d_rate_real = [0] * len(data.Time)
    drag = [0] * len(data.Time)
    thrust = [0] * len(data.Time)
    fuel_flow_bada = [0] * len(data.Time)

    #climb
    climb_drag = []
    climb_thrust = []
    climb_fuel_flow_bada = []
    climb_fuel_flow = []
    cl_rate_sections = []
    cl_rate_sections_real = []
    #cruise
    cruise_drag = []
    cruise_thrust = []
    cruise_fuel_flow_bada = []
    cruise_fuel_flow = []
    #descent
    descent_drag = []
    descent_thrust = []
    descent_fuel_flow_bada = []
    descent_fuel_flow = []
    d_rate_sections = []
    d_rate_sections_real = [] 
    #Creating necessary variables
    fuel_flow = (data.aFF1+data.aFF2)*constants["lb_to_kg"]*constants["hr_to_sec"]
    mass = data.aZFW*constants['lb_to_kg']+data.aTFQ-fuel_flow

    #utilityfunctions.py
    sigma = pressureRatio(data.aSAP)
    theta = temperatureRatio(data.aSAT)
    isaDev = isaDeviation(data.aALTBARO, data.aSAT)
    tas = cas2tas(data.aCAS, data.aALTBARO, isaDev)
    isa_pR = isaPressureRatio(data.aSAT, isaDev)
    isa_tC = isaTempC(data.aALTBARO, isaDev)
    tempC = temperatureC(data.aALTBARO, isaDev)
    tempK = tempC + 273.15
    deltaT = tempK - 288.15
    pres = isaPressure(tempK,isaDev)
    ccasesf = ac.constantCasESF(data.aMACH, isaDev, tempK)
    cmachesf = ac.constantMachESF(data.aMACH, isaDev, tempK)

    #Decomposing the fligt regimes
    for i in range(len(data)-1):
        flightMode, ROCD = ac.identifyFlightMode(data, i)
        if flightMode == 1 :
            cl_rate_real[i] = ROCD
            drag[i] = ac.calculate_drag(mass[i],sigma[i], data.aMACH[i],0)
            thrust[i] = ac.calculate_climb_performance(sigma[i], theta[i], data.aMACH[i], 0)[0]
            fuel_flow_bada[i] = ac.calculate_climb_performance(sigma[i], theta[i], data.aMACH[i], 0)[1]
            cl_rate[i] = ac.calculateROCD(thrust[i], drag[i], ccasesf[i], tas[i], mass[i], tempK[i], isaDev[i])
                       
            climb_drag.append(drag[i]) 
            climb_thrust.append(thrust[i])
            climb_fuel_flow_bada.append(fuel_flow_bada[i])
            climb_fuel_flow.append(fuel_flow[i]) 
            cl_rate_sections.append(cl_rate[i])
            cl_rate_sections_real.append(cl_rate_real[i])

        elif flightMode == 2 :
            drag[i] = ac.calculate_drag(mass[i],sigma[i],data.aMACH[i],0)
            thrust[i] = ac.calculate_cruise_performance(sigma[i], theta[i], data.aMACH[i], drag[i])[0]
            fuel_flow_bada[i] = ac.calculate_cruise_performance(sigma[i], theta[i], data.aMACH[i], drag[i])[1]

            cruise_drag.append(drag[i])
            cruise_thrust.append(thrust[i])
            cruise_fuel_flow_bada.append(fuel_flow_bada[i])
            cruise_fuel_flow.append(fuel_flow[i])

        elif flightMode == 3:
            d_rate_real[i] = ROCD
            drag[i] = ac.calculate_drag(mass[i],sigma[i],data.aMACH[i],0)
            thrust[i] = ac.calculate_descent_performance(sigma[i], theta[i], data.aMACH[i])[0]
            fuel_flow_bada[i] = ac.calculate_descent_performance(sigma[i], theta[i], data.aMACH[i])[1]
            d_rate[i] = ac.calculateROCD(thrust[i], drag[i], ccasesf[i], tas[i], mass[i], tempK[i], isaDev[i])

            descent_drag.append(drag[i])
            descent_thrust.append(thrust[i])
            descent_fuel_flow_bada.append(fuel_flow_bada[i])
            descent_fuel_flow.append(fuel_flow[i])
            d_rate_sections.append(d_rate[i])
            d_rate_sections_real.append(d_rate_real[i])

    #Adding BADA4 Columns to the data file
    data.insert(24, "BADA4 Thrust[N]", pd.Series(thrust), True)
    data.insert(25, "BADA4 Drag[N]", pd.Series(drag), True)
    data.insert(26, "BADA4 Total Fuel Flow[kg/s]", pd.Series(fuel_flow_bada), True)
    data.insert(27, "Total Fuel Flow[kg/s]", pd.Series(fuel_flow), True)
    data.insert(28, "BADA4 Rate of Climb", pd.Series(cl_rate), True)
    data.insert(29, "Rate of Climb", pd.Series(cl_rate_real), True)
    data.insert(30, "BADA4 Rate of Descent", pd.Series(d_rate), True)
    data.insert(31, "Rate of Descent", pd.Series(d_rate_real), True)
    data.to_csv("NewData/" + file, index=False)

    #Plotting and saving as PDF file

    #Thrust, Drag / Time
    fig11 = plt.figure(figsize=[7, 5])
    plt.plot(climb_drag, label="BADA4 Climb Drag",linewidth=.8)
    plt.plot(climb_thrust, label="BADA4 Climb Thrust",linewidth=.8)
    plt.title('Thrust-Drag graph', fontsize=12, y=1.1)
    plt.ylabel('Thrust-Drag (N)')
    plt.xlabel('Sample Point')
    plt.legend(loc="upper right")

    fig12 = plt.figure(figsize=[7, 5])
    plt.plot(cruise_drag, label="BADA4 Cruise Drag",linewidth=.8)
    plt.plot(cruise_thrust, label="BADA4 Cruise Thrust",linewidth=.8)
    plt.title('Thrust-Drag graph', fontsize=12, y=1.1)
    plt.ylabel('Thrust-Drag (N)')
    plt.xlabel('Sample Point')
    plt.legend(loc="upper right")

    fig13 = plt.figure(figsize=[7, 5])
    plt.plot(descent_drag, label="BADA4 Descent Drag",linewidth=.8)
    plt.plot(descent_thrust, label="BADA4 Descent Thrust",linewidth=.8)
    plt.title('Thrust-Drag graph', fontsize=12, y=1.1)
    plt.ylabel('Thrust-Drag (N)')
    plt.xlabel('Sample Point')
    plt.legend(loc="upper right")

    #Fuel Flow / Time
    fig21 = plt.figure(figsize=[7, 5])
    plt.plot(climb_fuel_flow_bada, label="BADA4 Climb Fuel Flow",linewidth=.8)
    plt.plot(climb_fuel_flow, label="Climb Fuel Flow",linewidth=.8)
    plt.title('Fuel Flow Graph', fontsize=12, y=1.1)
    plt.ylabel('Fuel Flow (kg/s)')
    plt.xlabel('Sample Point')
    plt.legend(loc="upper right")

    fig22 = plt.figure(figsize=[7, 5])
    plt.plot(cruise_fuel_flow_bada, label="BADA4 Cruise Fuel Flow",linewidth=.8)
    plt.plot(cruise_fuel_flow, label="Cruise Fuel Flow",linewidth=.8)
    plt.title('Fuel Flow Graph', fontsize=12, y=1.1)
    plt.ylabel('Fuel Flow (kg/s)')
    plt.xlabel('Sample Point')
    plt.legend(loc="upper right")

    fig23 = plt.figure(figsize=[7, 5])
    plt.plot(descent_fuel_flow_bada, label="BADA4 Descent Fuel Flow",linewidth=.8)
    plt.plot(descent_fuel_flow, label="Descent Fuel Flow",linewidth=.8)
    plt.title('Fuel Flow Graph', fontsize=12, y=1.1)
    plt.ylabel('Fuel Flow (kg/s)')
    plt.xlabel('Sample Point')
    plt.legend(loc="upper right")

    #ROCD
    fig3 = plt.figure(figsize=[7, 5])
    plt.plot(cl_rate_sections, label="BADA4 Rate of Climb",linewidth=.8)
    plt.plot(cl_rate_sections_real, label="Rate of Climb",linewidth=.8)
    plt.title('Rate of Climb Graph', fontsize=12, y=1.1)
    plt.ylabel('Rate of Climb (ft/s)')
    plt.xlabel('Sample Point')
    plt.legend(loc="upper right")

    fig4 = plt.figure(figsize=[7, 5])
    plt.plot(d_rate_sections, label="BADA4 Rate of Descent",linewidth=.8)
    plt.plot(d_rate_sections_real, label="Rate of Descent",linewidth=.8)
    plt.title('Rate of Descent Graph', fontsize=12, y=1.1)
    plt.ylabel('Rate of Descent (ft/s)')
    plt.xlabel('Sample Point')
    plt.legend(loc="lower right")

    pp = PdfPages("NewData/" + file +'figs.pdf')
    pp.savefig(fig11)
    pp.savefig(fig12)
    pp.savefig(fig13)
    pp.savefig(fig21)
    pp.savefig(fig22)
    pp.savefig(fig23)
    pp.savefig(fig3)
    pp.savefig(fig4)
    pp.close()




