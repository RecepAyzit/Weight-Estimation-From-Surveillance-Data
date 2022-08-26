from calendar import c
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

from scipy import optimize

def segments_fit(X, Y, count):
    X = np.array(X)
    Y = np.array(Y)
    xmin = X.min()
    xmax = X.max()

    seg = np.full(count - 1, (xmax - xmin) / count)

    px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
    py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.01].mean() for x in px_init])

    def func(p):
        seg = p[:count - 1]
        py = p[count - 1:]
        px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        return px, py

    def err(p):
        px, py = func(p)
        Y2 = np.interp(X, px, py)
        return np.mean((Y - Y2)**2)

    r = optimize.minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')
    return func(r.x)

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

specific_energy_mean = []
specific_energy_gradient_mean = []
takeoff_weight = []

basepath = "C:/Users/Recep/Desktop/Boeing Proje/"
odp = 'B777_Data'  ## Airport Pairs
datapath = basepath + odp

files = [f for f in listdir(datapath) if isfile(join(datapath, f))]

fig5 = plt.figure(5,figsize=[7, 5])
fig6 = plt.figure(6,figsize=[7, 5])
fig7 = plt.figure(7,figsize=[7, 5])
fig8 = plt.figure(8,figsize=[7, 5])
fig9 = plt.figure(9,figsize=[7, 5])

ac = JetAircraft('B773ERGE115B')
counter = 0
for file in files:
    counter +=1
    if counter == 1000:
        break
    filepath = os.path.join(odp, file)
    data = pd.read_csv(filepath, low_memory=False, skiprows=1)
    data.rename( columns={'Unnamed: 0':'Time'}, inplace=True )
    #Creating necessary variables
    fuel_flow = (data.aFF1+data.aFF2)*constants["lb_to_kg"]*constants["hr_to_sec"]
    #mass = data.aZFW*constants['lb_to_kg']+data.aTFQ-fuel_flow

    #utilityfunctions.py
    isaDev = isaDeviation(data.aALTBARO, data.aSAT)
    tas = cas2tas(data.aCAS, data.aALTBARO, isaDev)
    fuel_flow = (data.aFF1+data.aFF2)*constants["lb_to_kg"]*constants["hr_to_sec"]
    Time = data.Time
    
    ## Flight Path
    # takeoff_x, takeoff_y = Spherical2Planar(data.LATPOS[0], data.LONPOS[0])
    # landing_x, landing_y = Spherical2Planar(data.LATPOS[len(data)-1], data.LONPOS[len(data)-1])
    # flight_x, flight_y = Spherical2Planar(data.LATPOS, data.LONPOS)
    
    takeoff_lat = data.LATPOS[0]
    takeoff_lon = data.LONPOS[0]
    landing_lat = data.LATPOS[len(data)-1]
    landing_lon = data.LONPOS[len(data)-1]
    
    ### Lists to Store 
    mass = []
    distance_to_landing = 0
    distance_to_landing_list = []
    distance_from_takeoff = 0
    distance_from_takeoff_list = []
    landing_alt_list = []
    landing_tas_list = []
    takeoff_alt_list = []
    takeoff_tas_list = []
    approach_speed_mean = []
    landing_weight_estimation_mean = []
    landing_weight_estimation_percentage_mean = []
    specific_energy_gradient = []
    specific_energy = []
    specificEnergyDuringFlight = []
    takeoff = False
    landing = False
    
    fuel_consumption = 0 
    mass_0 = data.aZFW[0] * constants['lb_to_kg'] + data.aTFQ[0]
    fuel_flow = (data.aFF1+data.aFF2)*constants["lb_to_kg"]*constants["hr_to_sec"]
    for i in range(len(data)-1):
        flightMode, ROCD = ac.identifyFlightMode(data.aALTBARO[i+1],data.aALTBARO[i])
        fuel_consumption += fuel_flow[i]
        mass.append(mass_0 - fuel_consumption)

        # Landing Data Preparation
        if(data.aALTBARO[len(data)-2-i] - data.aALTBARO[len(data)-1-i] > 10):
            distance_to_landing += -haversine((data.LATPOS[len(data)-1-i],data.LONPOS[len(data)-1-i]),(data.LATPOS[len(data)-2-i],data.LONPOS[len(data)-2-i])) * constants["m_to_nm"]
            distance_to_landing_list.append(distance_to_landing)
            landing_alt_list.append(data.aALTBARO[len(data)-1-i])
            landing_tas_list.append(data.aCAS[len(data)-1-i])
            if -1.5 < distance_to_landing < -0.5: 
                approach_speed_mean.append(data.aCAS[len(data)-1-i] * knot_to_ms) 
                weightEstimation = ((data.aCAS[len(data)-1-i] - 5)/1.23 * knot_to_ms)**2 * rho0 * ac.wing_area * max(ac.cl_max_lgdn) * 0.5 / g0
                landing_weight_estimation_mean.append(weightEstimation)
                landing_weight_estimation_percentage_mean.append(weightEstimation / ac.reference_mass * 100)
                landing = True
         
        # Takeoff Data Preparation       
        if data.aALTBARO[i+1] - data.aALTBARO[i] > 10:
            takeoff_alt_list.append(data.aALTBARO[i])
            takeoff_tas_list.append(data.aCAS[i])
                
            distance_from_takeoff += haversine((data.LATPOS[i],data.LONPOS[i]),(data.LATPOS[i+1],data.LONPOS[i+1])) * constants["m_to_nm"]
            distance_from_takeoff_list.append(distance_from_takeoff)
            specificEnergyDuringFlight.append((tas[i]*knot_to_ms) ** 2 + g0 * data.aALTBARO[i] * ft_to_m)

            if 8 < distance_from_takeoff < 12:    # Distance to Takeoff
                specific_energy.append((tas[i]*knot_to_ms) ** 2 + g0 * data.aALTBARO[i] * ft_to_m)
                #specific_energy.append(g0 * data.aALTBARO[i] * ft_to_m)

                takeoff = True
    
    if takeoff and landing:
        print("Counter: ",counter)
        specific_energy_gradient = np.gradient(specific_energy)
        specific_energy_gradient_mean.append(sum(specific_energy_gradient))#/len(specific_energy_gradient))
        takeoff_weight.append(mass[0])
        specific_energy_mean.append(sum(specific_energy)/len(specific_energy))
        approach_speed.append(sum(approach_speed_mean)/len(approach_speed_mean))
        landing_weight_estimation.append(sum(landing_weight_estimation_mean)/len(landing_weight_estimation_mean))
        landing_weight_estimation_percentage.append(sum(landing_weight_estimation_percentage_mean)/len(landing_weight_estimation_percentage_mean))
        mtow_percentage.append((mass[i])/ac.reference_mass*100)
        
        #px, py = segments_fit(distance_to_takeoff[0:1000], data.aGS[0:1000], 2)
        plt.figure(5)
        plt.plot(distance_from_takeoff_list,takeoff_tas_list, linewidth=.5, color = 'k')
        plt.title('TAS vs Distance from Takeoff', fontsize=12, y=1.1)
        plt.xlim(0,30)
        plt.ylabel('TAS (kt)')
        plt.xlabel('Distance from Takeoff (NM)')
        plt.grid()
        
        plt.figure(6)
        plt.plot(distance_from_takeoff_list,specificEnergyDuringFlight, linewidth=.5, color = 'k')
        plt.title('Specific Energy vs Distance from Takeoff', fontsize=12, y=1.1)
        plt.xlim(0,30)
        plt.ylabel('Specific Energy (J/kg)')
        plt.xlabel('Distance from Takeoff (NM)')
        plt.grid()
        
        plt.figure(7)
        plt.plot(distance_from_takeoff_list,takeoff_alt_list, linewidth=.5, color = 'k')
        plt.title('Altitude vs Distance from Takeoff', fontsize=12, y=1.1)
        plt.xlim(0,30)
        plt.ylabel('Altitude (ft)')
        plt.xlabel('Distance from Takeoff (NM)')
        plt.grid()
        
        plt.figure(8)
        plt.plot(distance_to_landing_list,landing_alt_list, linewidth=.5, color = 'k')
        plt.title('Altitude vs Distance to Landing', fontsize=12, y=1.1)
        plt.xlim(-20,0)
        plt.ylabel('Altitude (ft)')
        plt.xlabel('Distance to Landing (NM)')
        plt.grid()
        
        plt.figure(9)
        plt.plot(distance_to_landing_list,landing_tas_list, linewidth=.5, color = 'k')
        plt.title('Distance to Landing vs TAS', fontsize=12, y=1.1)
        plt.xlim(-20,0)
        plt.ylabel('TAS')
        plt.xlabel('Distance to Landing (NM)')
        plt.grid()
                
plt.show()
         
#Linear regression for takeoff weight estimation from specific energy
A = np.vstack([specific_energy_mean, np.ones(len(specific_energy_mean))]).T
#A = np.vstack([specific_energy_gradient_mean, np.ones(len(specific_energy_gradient_mean))]).T
takeoff_weighty = np.array(takeoff_weight).reshape((-1,1))
takeoff_weight = np.array(takeoff_weight)
alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),takeoff_weighty)

print("Slope: ", alpha[0])
print("Constant: ", alpha[1])

#Take-off weight estimation parameters

takeoff_weight_estimation = alpha[0] * specific_energy_mean + alpha[1]
takeoff_weight_estimation_percentage = takeoff_weight_estimation / ac.reference_mass*100
takeoff_weight_percentage = takeoff_weight / ac.reference_mass*100

takeoff_weight_error = (takeoff_weight - takeoff_weight_estimation)/ac.reference_mass*100
mean_takeoff_weight_error = np.mean(takeoff_weight_error)
std_takeoff_weight_error = np.std(takeoff_weight_error)

### Data Visualization

fig1 = plt.figure(figsize=[7, 5])
x = range(0,200)
plt.plot(x,x, linewidth=.5, color = 'r')
plt.scatter(takeoff_weight_percentage, takeoff_weight_estimation_percentage, s = 2, color = 'k')
plt.title('Take-off Weight Estimation', fontsize=12, y=1.1)
plt.xlim(50, 100)
plt.ylim(50, 100)
plt.ylabel('MTOW(%) Estimated')
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

#x = range(0,150000)
x = range(0,200000)
fig10 = plt.figure(figsize=[7, 5])
plt.scatter(specific_energy_mean, takeoff_weight_percentage, s = 2, color = 'k')
#plt.scatter(specific_energy_gradient_mean, takeoff_weight_percentage, s = 2, color = 'k')
plt.plot(x,(alpha[0]*x + alpha[1]) / ac.reference_mass * 100, linewidth=1, color = 'k')
plt.title('Take-off Weight vs Specific Energy', fontsize=12, y=1.1)
plt.ylim(0, 100)
plt.ylabel('MTOW(%) Actual')
plt.xlabel('Specific Energy Mean')
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

pp = PdfPages('Results/' + odp + '/' + str(time.time()) + '.pdf')
pp.savefig(fig10)
pp.savefig(fig1)
pp.savefig(fig2)
pp.savefig(fig5)
pp.savefig(fig6)
pp.savefig(fig7)
pp.savefig(fig8)
pp.savefig(fig9)
pp.savefig(fig3)
pp.savefig(fig4)

#pp.savefig(fig10)
pp.close()