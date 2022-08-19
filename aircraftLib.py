import scipy.io
#from math import *
import numpy as np
import pandas as pd
import time
class JetAircraft:

    def __init__(self, ac_type):
        mat = scipy.io.loadmat(ac_type+'.mat', squeeze_me=True, struct_as_record=False)
        self.defaultFile = mat

        # Drag Parameters
        self.reference_mass = mat[ac_type].Model.PFM.MTOW
        self.wing_area = mat[ac_type].Model.AFM.S

        self.clean_drag = mat[ac_type].Model.AFM.DPM.d
        self.drag_scalar = mat[ac_type].Model.AFM.DPM.scalar

        self.non_clean_drag_lgup = mat[ac_type].Model.AFM.DPM.LGUP.d
        self.high_lift_values_lgup = mat[ac_type].Model.AFM.DPM.LGUP.HL
        self.cl_max_lgup = mat[ac_type].Model.AFM.DPM.LGUP.CL_max

        self.non_clean_drag_lgdn = mat[ac_type].Model.AFM.DPM.LGDN.d
        self.high_lift_values_lgdn = mat[ac_type].Model.AFM.DPM.LGDN.HL
        self.cl_max_lgdn = mat[ac_type].Model.AFM.DPM.LGDN.CL_max

        # Thrust Parameters
        self.idle_thrust = mat[ac_type].Model.PFM.PPM.LIDL.ti
        self.generalized_thrust = mat[ac_type].Model.PFM.PPM.a
        self.mcmb_thrust = mat[ac_type].Model.PFM.PPM.MCMB.b
        self.mcmb_thrust_2 = mat[ac_type].Model.PFM.PPM.MCMB.c
        self.throttle_max = mat[ac_type].Model.PFM.PPM.throttle.high
        self.throttle_min = mat[ac_type].Model.PFM.PPM.throttle.low

        # Fuel Parameters
        self.LHV = mat[ac_type].Model.PFM.LHV
        self.generalized_fuel = mat[ac_type].Model.PFM.PPM.f
        self.idle_fuel = mat[ac_type].Model.PFM.PPM.LIDL.fi

        # Atmospheric parameters
        self.T0 = 288.15
        self.gas_constant = 287.05
        self.adiabatic_index = 1.4
        self.p0 = 101325
        self.rho0 = 1.225
        self.a0 = 340.294
        self.g0 = 9.80665
        self.beta = -0.0065

        # Flight Status
        self.in_climb = 1
        self.in_cruise = 0
        self.in_descent = -1
        self.flight_status = 0

        # States
        self.vcasKt = 0
        self.vtasKt = 0
        self.altitudeFt = 0
        self.mach = 0
        self.x_position = 0
        self.y_position = 0

    def calculate_drag(self, mass, sigma, mach, bank):
        L = mass * self.g0 / np.cos(bank)
        cl = 2*L / (sigma * self.p0 * self.adiabatic_index * self.wing_area * mach**2)
        fm = 1 / (np.sqrt(1-mach**2))
        d = self.clean_drag
        cd1 = d[0] + d[1]*fm + d[2]*(fm**2) + d[3]*(fm**3) + d[4]*(fm**4)
        cd2 = (d[5] + d[6]*(fm**3) + d[7]*(fm**6) + d[8]*(fm**9) + d[9]*(fm**12))*(cl**2)
        cd3 = (d[10] + d[11]*(fm**14) + d[12]*(fm**15) + d[13]*(fm**16) + d[14]*(fm**17))*(cl**6)
        cd = (cd1 + cd2 + cd3)*self.drag_scalar
        drag = L * cd / cl
        return drag

    def calculate_lift_coefficient(self, mass, sigma, mach):
        cl = 2*mass*self.g0 / (sigma * self.p0 * self.adiabatic_index * self.wing_area * mach**2)
        return cl
    
    def calculate_non_clean_drag(self, mass, sigma, mach, flap, speed_break, landing_gear):
        L = mass * self.g0
        cl = 2*L / (sigma * self.p0 * self.adiabatic_index * self.wing_area * mach**2)
        
        if landing_gear == 0:
            d = self.non_clean_drag_lgup
            
            flap_index = (np.abs(self.high_lift_values_lgup - flap)).argmin()
            
            cd = d[flap_index, 0] + d[flap_index, 1]*cl + d[flap_index, 2] * cl**2
            
        else:
            d = self.non_clean_drag_lgdn
            
            flap_index = (np.abs(self.high_lift_values_lgdn - flap)).argmin()
            
            cd = d[flap_index, 0] + d[flap_index, 1]*cl + d[flap_index, 2] * cl**2
            
        if speed_break != 0:
            cd = cd + 0.03
            
        drag = L * cd / cl
        
        return drag

    def calculate_descent_performance(self, sigma, theta, mach):

        # Low idle thrust
        ti = self.idle_thrust
        ct = 0
        for i in range(3):
            for j in range(4):
                ct = ct + ti[4*i+j] * (sigma**(j-1)) * (mach**i)

        thrust = self.reference_mass * self.g0 * sigma * ct

        # Low idle fuel consumption
        fi = self.idle_fuel
        cf = 0
        for i in range(3):
            for j in range(3):
                cf = cf + fi[3*i+j] * (sigma**j) * (mach**i)

        cf = cf * sigma**(-1) * theta**(-0.5)
        fuel_flow = sigma * theta**0.5 * self.reference_mass * self.g0 * self.a0 * cf / self.LHV

        return thrust, fuel_flow

    def calculate_cruise_performance(self, sigma, theta, mach, drag):

        # Cruise Thrust
        thrust = drag
        ct = thrust / (self.reference_mass * self.g0 * sigma)

        # Cruise Fuel Consumption
        f = self.generalized_fuel
        cf = 0
        for i in range(5):
            for j in range(5):
                cf = cf + f[5*i+j] * ct**j * mach**i

        fuel_flow = sigma * theta**0.5 * self.reference_mass * self.g0 * self.a0 * cf / self.LHV
        return thrust, fuel_flow

    def calculate_climb_performance(self, sigma, theta, mach, deltaT):

        # Climb Throttle
        if np.abs(deltaT) < 12:

            b = self.mcmb_thrust
            st = 0
            for i in range(6):
                for j in range(6):
                    st = st + b[6*i+j] * mach**j * sigma**i
                
        else:
            b = self.mcmb_thrust_2
            thetaT = theta * (1 + 0.2*(mach**2))
            st = 0
            for i in range(6):
                for j in range(6):
                    st = st + b[6*i+j] * mach**j * thetaT**i

        # Climb Thrust
        a = self.generalized_thrust
        ct = 0
        for i in range(6):
            for j in range(6):
                ct = ct + a[6*i+j] * mach**j * st**i

        thrust = self.reference_mass * self.g0 * sigma * ct

        # Climb Fuel Consumption
        f = self.generalized_fuel
        cf = 0
        for i in range(5):
            for j in range(5):
                cf = cf + f[5*i+j] * ct**j * mach**i

        fuel_flow = sigma * theta ** 0.5 * self.reference_mass * self.g0 * self.a0 * cf / self.LHV

        return thrust, fuel_flow

    def climb_correction_factor(self, pDataFrame):
        # Inputs
        alt_ft = pDataFrame['aALTBARO'].values
        tempK = pDataFrame['aSAT'].values + 273.0
        mass = pDataFrame['Mass (kg)'].values
        mach = pDataFrame['Mach'].values
        sigma = pDataFrame["Pressure Ratio"].values 
        flap = pDataFrame['Flap (deg)'].values
        speed_break = pDataFrame['Speed Break'].values
        landing_gear = pDataFrame['Landing Gear'].values
        isaDev = pDataFrame["ISA Deviation (C)"].values

        # Global Constants
        kappa = 1.4
        R = 287.05287
        g0 = 9.80665
        lb_to_kg = 0.45359
        hr_to_sec = 1/3600

        # BADA4 Fuel Rate

        # Temperature Ratio
        theta = tempK / 288.15

        
        climb_T = []
        climb_F = []
        for i in range(len(alt_ft)):
            T, F = self.calculate_climb_performance(sigma[i], theta[i], mach[i], isaDev[i])
            climb_T.append(T)
            climb_F.append(F)
        print('Thrust is calculated')

        #climb_T, climb_F = self.calculate_climb_performance(sigma, theta, mach, isaDev)

        pDataFrame["BadaFuelFlow"] = climb_F
        pDataFrame["TotalFuelFlow"] = (pDataFrame['FF1 (lb/hr)'].values + 
                                    pDataFrame['FF2 (lb/hr)'].values) * lb_to_kg * hr_to_sec
        
        pDataFrame["CorrectionFactor"] = np.divide( pDataFrame["TotalFuelFlow"], pDataFrame["BadaFuelFlow"] )
        
        return pDataFrame

    def cruise_correction_factor(self, pDataFrame):
        # Inputs
        alt_ft = pDataFrame['Altitude (ft)'].values
        tempK = pDataFrame['Temp (C)'].values + 273.0
        mass = pDataFrame['Mass (kg)'].values
        mach = pDataFrame['Mach'].values
        sigma = (1-2.25577*10**-5*(pDataFrame['Altitude (ft)'].values * 0.3048))**5.25588
        sigma = pDataFrame["Air Pressure (mb)"].values / 1013.25
        flap = pDataFrame['Flap (deg)'].values
        speed_break = pDataFrame['Speed Break'].values
        landing_gear = pDataFrame['Landing Gear'].values

        # Global Constants
        kappa = 1.4
        R = 287.05287
        g0 = 9.80665
        lb_to_kg = 0.45359
        hr_to_sec = 1/3600

        # BADA4 Fuel Rate
        h_meter = alt_ft * 0.3048

        # Temperature Ratio
        theta = tempK / 288.15

        cmb_drag = np.zeros((1,1))
        cmb_thr = np.zeros((1,1))
        cmb_F = np.zeros((1,1))

        Drag = self.calculate_drag( mass, sigma, mach, 0 )     
        print('Drag is calcuated')

        cruise_T, cruise_F = self.calculate_cruise_performance(sigma, theta, mach, Drag)
        print('Thrust is calculated')

        # Fuel Flow of QAR
        pDataFrame["BadaFuelFlow"] = cruise_F
        pDataFrame["TotalFuelFlow"] = (pDataFrame['FF1 (lb/hr)'].values + pDataFrame['FF2 (lb/hr)'].values) * lb_to_kg * hr_to_sec
        pDataFrame["CorrectionFactor"] = np.divide( pDataFrame["TotalFuelFlow"], pDataFrame["BadaFuelFlow"] )
        
        return pDataFrame

    def descent_correction_factor(self, pDataFrame):
        # Inputs
        alt_ft = pDataFrame['Altitude (ft)'].values
        tempK = pDataFrame['Temp (C)'].values + 273.0
        mass = pDataFrame['Mass (kg)'].values
        mach = pDataFrame['Mach'].values
        sigma = (1-2.25577*10**-5*(pDataFrame['Altitude (ft)'].values * 0.3048))**5.25588
        flap = pDataFrame['Flap (deg)'].values
        speed_break = pDataFrame['Speed Break'].values
        landing_gear = pDataFrame['Landing Gear'].values

        # Global Constants
        kappa = 1.4
        R = 287.05287
        g0 = 9.80665
        lb_to_kg = 0.45359
        hr_to_sec = 1/3600

        # BADA4 Fuel Rate
        h_meter = alt_ft * 0.3048

        # Temperature Ratio
        theta = tempK / 288.15

        cmb_drag = np.zeros((1,1))
        cmb_thr = np.zeros((1,1))
        cmb_F = np.zeros((1,1))

        climb_T, descent_F = self.calculate_descent_performance(sigma, theta, mach)
        print('Thrust is calculated')

        pDataFrame["BadaFuelFlow"] = descent_F
        pDataFrame["TotalFuelFlow"] = (pDataFrame['FF1 (lb/hr)'].values + pDataFrame['FF2 (lb/hr)'].values) * lb_to_kg * hr_to_sec
        pDataFrame["CorrectionFactor"] = np.divide( pDataFrame["TotalFuelFlow"], pDataFrame["BadaFuelFlow"] )
        
        return pDataFrame

    def constantMachESF(self, mach, isaDev, tempK):
        
        fM = 1+self.adiabatic_index*self.gas_constant*self.beta*(mach**2)*((tempK-isaDev)/tempK)/(2*self.g0)
        return 1/fM

    def constantCasESF(self, mach, isaDev, tempK):
        a1 = (1 + (self.adiabatic_index-1)*(mach**2)/2)
        a2 = (a1**((self.adiabatic_index)/(self.adiabatic_index-1)))-1
        a3 = a1**(-1/(self.adiabatic_index-1))
        a4 = a2*a3 + 1 + (tempK - isaDev)*(mach**2)*(self.adiabatic_index*self.gas_constant*self.beta)/(2*self.g0*tempK)
        return 1/a4

    def calculateROCD(self, thrust, drag, esf, tas, mass, tempK, isadev):
        rocd = (tempK-isadev) * (thrust - drag) * tas * esf / (mass * self.g0 * tempK)
        return rocd

    def identifyFlightMode(self,data,i):

        flightMode = 0
        ROCD = data.aALTBARO[i+1] - data.aALTBARO[i] 

        if np.abs(ROCD) < 10:
            ROCD = 0

        if ROCD > 0 :
            flightMode = 1
        
        elif ROCD == 0 :
            flightMode = 2
        
        elif ROCD < 0 :
            flightMode = 3
        
        else:
            flightMode = 0

        return flightMode, ROCD

ac = JetAircraft( 'B773ERGE115B' )

