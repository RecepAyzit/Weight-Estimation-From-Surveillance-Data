import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
from math import sin, cos, atan2, sqrt, radians, asin, floor
import math
from torch import long

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
constants["knot_to_meters"] = 0.51444

kappa = constants["kappa"]
mu = constants["mu"]
a0 = constants["a0"]
beta = constants["beta"]
g0 = constants["g0"]
R = constants["R"]
T0 = constants["T0"]
p0 = constants["p0"]
rho0 = constants["rho0"]

def tas2cas(tasKt, altFt, isaDev):
    
    p0 = 101325
    sigma = (1-2.25577*10**-5*(altFt * 0.3048))**5.25588
    
    tempK = 288.15 - 6.5*(altFt * 0.3048)/1000 + isaDev
    rho = sigma * 101325 / (287.05287 * tempK)
    rho0 = 1.225
    p = p0 * sigma
    
    tasMps = tasKt * 0.5144

    mu = 1 / 3.5
    c0 = (1 + (mu * rho * tasMps**2) / (2 * p)) ** (1/mu)
    c1 = (c0 - 1) * sigma + 1
    c2 = c1**mu - 1
    c3 = 2 * p0 * c2 / (mu * rho0)
    
    casMps = np.sqrt( c3 )
    casKt = casMps / 0.5144
    
    return casKt

def cas2tas(pCasKt, pAltFeet, pIsaDev):
    pCasMps = pCasKt * 0.5144
    tempK = temperatureC(pAltFeet, pIsaDev) + 273
    p = isaPressure(tempK, pIsaDev)
    rho = p / (R * tempK)
    VtasMps = (1 + mu * 0.5 * (rho0 / p0) * pCasMps**2)**(1/mu) - 1
    VtasMps = (1 + (p0/p) * VtasMps)**mu - 1
    VtasMps = ((2/mu) * (p/rho) * VtasMps)**0.5
    
    VtasKt = VtasMps / 0.5144
    return VtasKt

def mach2tasKt(pMach, pAltFeet, pIsaDev):
    tempK = temperatureC(pAltFeet, pIsaDev) + 273
    pTasKt = (np.sqrt( kappa * R * tempK )) * pMach / 0.5144
    return pTasKt

def tas2mach(pTasMps, pAltFeet, pIsaDev):
    tempK = temperatureC(pAltFeet, pIsaDev) + 273
    pMach = pTasMps / (np.sqrt( kappa * R * tempK ))
    return pMach

def isaPressure(pTempK, pIsaDev): 
    pres = p0 * ((pTempK - pIsaDev)/T0)**(-g0/(beta * R))
    return pres

def pressureRatio(pDataFrame):
    pRatio = pDataFrame / 1013.25
    pDataFrame = pRatio
    return pRatio

def isaPressureRatio(pTempC, pIsaDev):
    pTempK = pTempC + 273
    pres = p0 * ((pTempK - pIsaDev)/T0)**(-g0/(beta * R))
    return pres / 101325

def temperatureRatio(pDataFrame):
    tRatio = (pDataFrame + 273)/288.15
    pDataFrame = tRatio
    return tRatio

def isaDeviation(pAltFeet, pTempC):
    isaTemp = (288.15 - 6.5*(pAltFeet * 0.3048)/1000) - 273
    isaDev = pTempC - isaTemp
    return isaDev

def isaTempC(pAltFeet, pIsaDev):
    tempK = 288.15 + beta * (pAltFeet * 0.3048) + pIsaDev
    tempC = tempK - 273
    return tempC

def temperatureC(pAltFeet, pIsaDev):
    pAltM = pAltFeet * 0.3048
    TempC = T0 + pIsaDev + beta * pAltM -273
    return TempC

def euclideanDistance(x_0, y_0, x, y):
    euclideanDistance = np.sqrt((x_0-x)**2 + (y_0-y)**2)
    return euclideanDistance

def Spherical2Planar(lat,lon):
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    R = 6371000
    x = R * np.cos(lat) * lon
    y = R * lat
    return x, y

def Spherical2Cartesian(latitude, longitude, elevation = 0):
    latitude = np.deg2rad(latitude)
    longitude = np.deg2rad(longitude)
    R = 6378137.0 + elevation  # relative to centre of the earth
    X = R * np.cos(longitude) * np.sin(latitude)
    Y = R * np.sin(longitude) * np.sin(latitude)
    Z = R * np.cos(latitude)

    return X,Y

def Pressure2Altitude(pressure, p0 = 101325):
    # Pressure is in Pascal
    alt = (pow(pressure/p0,1/5.25588) - 1)/(-0.0000225577)
    return alt

def haversine(point1, point2):
    """ Calculate the great-circle distance between two points on the Earth surface.

    Takes two 2-tuples, containing the latitude and longitude of each point in decimal degrees,
    and, optionally, a unit of length.

    :return: the distance between the two points in meters, as a float.
    """
    earth_radius = 6371008.8 # meters

    # unpack latitude/longitude
    lat1, lng1 = point1
    lat2, lng2 = point2

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1 = np.deg2rad(lat1)
    lng1 = np.deg2rad(lng1)
    lat2 = np.deg2rad(lat2)
    lng2 = np.deg2rad(lng2)

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    return 2 * earth_radius * np.arcsin(np.sqrt(d))

def WeightFromSpeed(pAltFeet,pIsaDev, S , Clmax, Vapp):
    tempK = temperatureC(pAltFeet, pIsaDev) + 273
    p = isaPressure(tempK, pIsaDev)
    rho = p / (R * tempK)
    print(max(Vapp), np.mean(Vapp))
    weight = (Vapp - 7/1.23)**2 * 0.5 * rho * S * Clmax * constants["knot_to_meters"]**2
    print(max(weight), np.mean(weight))
    return weight

def Tas2Ias(Tas,a0,rho,P0,Ki=0):
    Ias = a0 * sqrt(5*(pow(0.5*rho*Tas*Tas/P0 + 1,2/7)-1)) + Ki
    return Ias
    
def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
