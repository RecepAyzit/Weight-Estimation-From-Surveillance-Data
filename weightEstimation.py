from turtle import color
import pandas as pd
from aircraftLib import JetAircraft
from UtilityFunctions import *
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import os
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression
from matplotlib import style
import time
style.use('ggplot')

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
constants["m_to_nm"] = 0.0005399568
constants["fts_to_knot"] = 0.592483801
constants["mph_to_knot"] = 0.868976242 
constants["knot_to_ms"] = 0.51444
constants["pTotal"] = 1013.25
constants["ft_to_m"] = 0.3048 

kappa = constants["kappa"]
mu = constants["mu"]
a0 = constants["a0"]
beta = constants["beta"]
g0 = constants["g0"]
R = constants["R"]
T0 = constants["T0"]
p0 = constants["p0"]
rho0 = constants["rho0"]
pTotal = constants["pTotal"]
knot_to_ms = constants["knot_to_ms"]
ft_to_m = constants["ft_to_m"]

mtow_percentage = []
approach_speed = []
landing_weight_estimation = []
landing_weight_estimation_percentage = []

specific_energy = []
takeoff_weight = []
datapath = "C:/Users/Recep/Desktop/Boeing Proje/B777_Data"
files = [f for f in listdir(datapath) if isfile(join(datapath, f))]

fig5 = plt.figure(5,figsize=[7, 5])
fig6 = plt.figure(6,figsize=[7, 5])
fig7 = plt.figure(7,figsize=[7, 5])


ac = JetAircraft('B773ERGE115B')
counter = 0
for file in files:
    counter +=1
    """ if counter == 10:
        break """
    filepath = os.path.join("B777_Data", file)
    data = pd.read_csv(filepath, low_memory=False, skiprows=1)
    data.rename( columns={'Unnamed: 0':'Time'}, inplace=True )
    #Creating necessary variables
    fuel_flow = (data.aFF1+data.aFF2)*constants["lb_to_kg"]*constants["hr_to_sec"]
    mass = data.aZFW*constants['lb_to_kg']+data.aTFQ-fuel_flow

    #utilityfunctions.py
    #sigma = pressureRatio(data.aSAP)
    #theta = temperatureRatio(data.aSAT)
    isaDev = isaDeviation(data.aALTBARO, data.aSAT)
    tas = cas2tas(data.aCAS, data.aALTBARO, isaDev)
    #isa_pR = isaPressureRatio(data.aSAT, isaDev)
    #isa_tC = isaTempC(data.aALTBARO, isaDev)
    #tempC = temperatureC(data.aALTBARO, isaDev)
    #tempK = tempC + 273.15
    #deltaT = tempK - 288.15
    #pres = isaPressure(tempK,isaDev)
    #ccasesf = ac.constantCasESF(data.aMACH, isaDev, tempK)
    #cmachesf = ac.constantMachESF(data.aMACH, isaDev, tempK)
    fuel_flow = (data.aFF1+data.aFF2)*constants["lb_to_kg"]*constants["hr_to_sec"]
    mass = data.aZFW*constants['lb_to_kg']+data.aTFQ-fuel_flow
    #mtow_percentage.append((mass[len(mass)-1])/ac.reference_mass*100)

    takeoff_x, takeoff_y = Spherical2Planar(data.LATPOS[0], data.LONPOS[0])
    landing_x, landing_y = Spherical2Planar(data.LATPOS[len(data)-1], data.LONPOS[len(data)-1])
    flight_x, flight_y = Spherical2Planar(data.LATPOS, data.LONPOS)
    
    takeoff_lat = data.LATPOS[0]
    takeoff_lon = data.LONPOS[0]
    landing_lat = data.LATPOS[len(data)-1]
    landing_lon = data.LONPOS[len(data)-1]
    
    
    distance_to_landing = []
    distance_to_takeoff = []
    
    ias_descent = []
    ias_climb = []
    alt_descent = []
    alt_climb = []
    approach_speed_mean = []
    landing_weight_estimation_mean = []
    landing_weight_estimation_percentage_mean = []
    specific_energy_mean = []
    specificEnergyDuringFlight = []
    takeoff = False
    landing = False

    for i in range(len(data)-1):

        flightMode, ROCD = ac.identifyFlightMode(data, i)
        #distance_to_takeoff.append(euclideanDistance(takeoff_x, takeoff_y, flight_x[i], flight_y[i]) * constants["m_to_nm"])
        distance_to_takeoff.append(haversine((takeoff_lat,takeoff_lon),(data.LATPOS[i],data.LONPOS[i])) * constants["m_to_nm"])
        specificEnergyDuringFlight.append((tas[i]*knot_to_ms) ** 2 + g0 * data.aALTBARO[i] * ft_to_m)
        #landing weight estimation
        if flightMode == 3:
            alt_descent.append(data.aALTBARO[i])
            ias_descent.append(data.aCAS[i])
            #distance_to_landing.append(-euclideanDistance(landing_x, landing_y, flight_x[i], flight_y[i]) * constants["m_to_nm"])
            distance_to_landing.append(-haversine((landing_lat,landing_lon),(data.LATPOS[i],data.LONPOS[i])) * constants["m_to_nm"])
            if -5 < -haversine((landing_lat,landing_lon),(data.LATPOS[i],data.LONPOS[i])) * constants["m_to_nm"] < -1:         
            #if -5 < -euclideanDistance(landing_x, landing_y, flight_x[i], flight_y[i]) * constants["m_to_nm"] < -1: 
                approach_speed_mean.append(data.aCAS[i] * knot_to_ms) 
                landing_weight_estimation_mean.append(((data.aCAS[i] - 7)/1.23 * knot_to_ms)**2 * rho0 * ac.wing_area * max(ac.cl_max_lgdn) * 0.5 / g0)
                landing_weight_estimation_percentage_mean.append(((data.aCAS[i] - 7)/1.23 * knot_to_ms)**2 * rho0 * ac.wing_area * max(ac.cl_max_lgdn) * 0.5 / ac.reference_mass * 100 / g0)
                landing = True
        #take-off weight estimation
        elif flightMode == 1:
            alt_climb.append(data.aALTBARO[i])
            ias_climb.append(data.aCAS[i])
            #distance_to_takeoff.append(euclideanDistance(takeoff_x, takeoff_y, flight_x[i], flight_y[i]) * constants["m_to_nm"])
            #distance_to_takeoff.append(haversine((takeoff_lat,takeoff_lon),(data.LATPOS[i],data.LONPOS[i])) * constants["m_to_nm"])
            if 8 < haversine((takeoff_lat,takeoff_lon),(data.LATPOS[i],data.LONPOS[i])) * constants["m_to_nm"] < 12:        
            #if 98 < euclideanDistance(takeoff_x, takeoff_y, flight_x[i], flight_y[i]) * constants["m_to_nm"] < 102:
                specific_energy_mean.append((tas[i]*knot_to_ms) ** 2 + g0 * data.aALTBARO[i] * ft_to_m)
                takeoff = True
    
    if takeoff and landing:
        takeoff_weight.append(mass[0])
        specific_energy.append(sum(specific_energy_mean)/len(specific_energy_mean))
        approach_speed.append(sum(approach_speed_mean)/len(approach_speed_mean))
        landing_weight_estimation.append(sum(landing_weight_estimation_mean)/len(landing_weight_estimation_mean))
        landing_weight_estimation_percentage.append(sum(landing_weight_estimation_percentage_mean)/len(landing_weight_estimation_percentage_mean))
        mtow_percentage.append((mass[i])/ac.reference_mass*100)
        
        plt.figure(5)
        plt.plot(distance_to_takeoff,tas[0:-1], linewidth=.5, color = 'k')
        #plt.plot(distance_to_takeoff[-1000:-1],tas[-1000:-1], linewidth=.5, color = 'k')
        plt.title('TAS vs Distance', fontsize=12, y=1.1)
        plt.ylabel('TAS (kt)')
        plt.xlabel('Distance (NM)')
        plt.grid()
        
        plt.figure(6)
        plt.plot(distance_to_takeoff,specificEnergyDuringFlight, linewidth=.5, color = 'k')
        plt.title('Specific Energy vs Distance', fontsize=12, y=1.1)
        plt.ylabel('Specific Energy (J/kg)')
        plt.xlabel('Distance (NM)')
        plt.grid()
        
        plt.figure(7)
        plt.plot(distance_to_takeoff,data.aALTBARO[0:-1], linewidth=.5, color = 'k')
        #plt.plot(distance_to_takeoff[-1000:-1],data.aALTBARO[-1000:-1], linewidth=.5, color = 'k')
        plt.title('Altitude vs Distance', fontsize=12, y=1.1)
        plt.ylabel('Altitude (ft)')
        plt.xlabel('Distance (NM)')
        plt.grid()
                

plt.show()
#Linear regression for takeoff weight estimation from specific energy
A = np.vstack([specific_energy, np.ones(len(specific_energy))]).T
takeoff_weighty = np.array(takeoff_weight).reshape((-1,1))
takeoff_weight = np.array(takeoff_weight)
alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),takeoff_weighty)

#Take-off weight estimation parameters

takeoff_weight_estimation = alpha[0]*specific_energy + alpha[1]
takeoff_weight_estimation_percentage = takeoff_weight_estimation/ac.reference_mass*100
takeoff_weight_percentage = takeoff_weight/ac.reference_mass*100

takeoff_weight_error = (takeoff_weight - takeoff_weight_estimation)/ac.reference_mass*100
mean_takeoff_weight_error = np.mean(takeoff_weight_error)
std_takeoff_weight_error = np.std(takeoff_weight_error)

fig1 = plt.figure(figsize=[7, 5])
x = range(0,200)
plt.plot(x,x, linewidth=.5, color = 'r')
plt.scatter(takeoff_weight_percentage, takeoff_weight_estimation_percentage, s = 2, color = 'k')
plt.title('Take-off Weight Estimation', fontsize=12, y=1.1)
plt.xlim(50, 100)
plt.ylim(50, 100)
plt.ylabel('MTOW(%) Calculated')
plt.xlabel('MTOW(%) Actual')
plt.grid()

fig2, ax2 = plt.subplots()
plt.hist(takeoff_weight_error, bins=26, edgecolor = "black")
plt.title('Take-off Weight Estimation Error', fontsize=12, y=1.1)
plt.ylabel('Number of Aircraft')
plt.xlabel('MTOW(%) Error')
props = dict(boxstyle='square', facecolor='cyan', alpha=0.5)
textstr =  f'Mean Absolute Error: {np.abs(mean_takeoff_weight_error):.2f}%MTOW\n Standart Deviation:{std_takeoff_weight_error:.2f}%MTOW'
plt.text(0.05, 0.95, textstr, transform = ax2.transAxes, fontsize = 8, verticalalignment ='top', bbox = props)
plt.xlim(-40, 40)

x = range(0,500000)
fig22 = plt.figure(figsize=[7, 5])
plt.scatter(specific_energy, takeoff_weight_percentage, s = 2, color = 'k')
plt.plot(x,(alpha[0]*x + alpha[1]) / ac.reference_mass * 100, linewidth=1, color = 'k')
plt.title('Take-off Weight Estimation', fontsize=12, y=1.1)
#plt.ylim(0, 100)
plt.ylabel('MTOW(%) Calculated')
plt.xlabel('Specific Energy')
plt.grid()
#landing weight estimation parameters

landing_weight_error = (landing_weight_estimation - mass[len(mass)-1]) / ac.reference_mass * 100
mean_landing_weight_error = np.mean(landing_weight_error)
std_landing_weight_error = np.std(landing_weight_error)
    #plot
fig3 = plt.figure(figsize=[7, 5])
plt.scatter(mtow_percentage, landing_weight_estimation_percentage, s = .5, color = 'k')
x = range(0,100)
plt.title('Landing Weight Estimation', fontsize=12, y=1.1)
plt.ylabel('MTOW(%) Calculated')
plt.xlabel('MTOW(%) Actual')
plt.plot(x,x, linewidth=.5, color = 'r')
plt.grid()
plt.xlim(0, 100)
plt.ylim(0, 100)

fig4, ax4 = plt.subplots()
plt.hist(landing_weight_error, bins=26, edgecolor = "black")
plt.ylabel('Number of Aircraft')
plt.xlabel('MTOW(%) Error')
textstr =  f'Mean Absolute Error: {np.abs(mean_landing_weight_error):.2f}%MTOW\n Standart Deviation:{std_landing_weight_error:.2f}%MTOW'
plt.text(0.05, 0.95, textstr, transform = ax4.transAxes, fontsize = 8, verticalalignment ='top', bbox = props)
plt.xlim(-40, 40)


pp = PdfPages("Weight Estimation/" + str(time.time()) + '.pdf')
pp.savefig(fig1)
pp.savefig(fig2)
pp.savefig(fig3)
pp.savefig(fig4)
pp.savefig(fig5)
pp.savefig(fig6)
pp.savefig(fig7)
pp.savefig(fig22)
pp.close()