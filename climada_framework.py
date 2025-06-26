# Copied from https://www.dropbox.com/home/CLIMADA_user_guidance-main
# Using my cloned fork of the CLIMADA GitHub repository with netCDF functionality
# https://github.com/babichmorrowc/climada_python

# Applies the CLIMADA risk assessment tool

# Datasets used here as example:
# Exposure: outdoor working population from SSP2
# Hazard: UKCP18 (12km) humidex (derived from temperature and humidity)
# Vulnerability: function taken from Foster et al. 2021
# Risk: annual expected work days lost

# conda env: ~/.conda/envs/climada_2024

import warnings
import sys
import os
import glob
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import colors
import cartopy.crs as ccrs
from pathlib import Path

##### Set home dir and import climada functions #####
HOME_DIR = Path.home()
# this is where the climada code is saved
sys.path.append(os.path.join(HOME_DIR, 'Documents/climada_python'))
# import climada functions
from climada.hazard import Hazard
from climada.entity import Exposures
from climada.entity import ImpactFunc, ImpactFuncSet
from climada.entity import Entity
from climada.engine import Impact
from climada.entity import Measure, MeasureSet
from climada.engine import CostBenefit
from climada.engine.cost_benefit import risk_aai_agg
from climada.entity import DiscRates
from climada.engine.unsequa import CalcImpact, InputVar #CLIMADA >= 3.1.0

### Set filepaths and custom number of years ###
DATA_DIR = 'data/'
# OUT_DIR = os.path.join(HOME_DIR, 'CLIMADA_user_guidance', 'output/')
OUT_DIR = os.path.join(HOME_DIR, 'Documents/climada_test/output/')
## Make this directory if it doesn't already exist
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# custom number of years if needed
CUSTOM_NYEARS = 360-102-8-25 #360 days - weekends - BHs - AL


def define_hazard(file_name, nc1, variable, haz_type, custom_nyears=False):
    """
    Define the hazard data and read it in from netcdf files to create a Heat
    stress  instance of a hazard object

    Inputs
    ------
        file_name: (string) name of netcdf file to be read in
        nc1: Dataset of hazard data
        variable: variable of interest in the netcdf file
        haz_type: user defined hazard category
        custom_nyears: set to True if want to alter number of years, e.g. to
            remove weekends or holidays

    Returns
    -------
         hazard: (Hazard Object)
    """
    nyears = round(nc1.dimensions['time'].size)/ 360 #climate day with 360 days per year
    if custom_nyears:
        nyears = (360*nyears)/CUSTOM_NYEARS
        print(f'using custom number of years: {nyears}')

    # Variables that help defined the Heat stress instance
    hazard_args = {'intensity_var': variable,
                   'event_id': np.arange(1, len(nc1.variables['time'])+1),
                   'frequency': np.full(len(nc1.variables['time']), 1/nyears),
                   'haz_type': haz_type,
                   'description': 'Hazard data',
                   'replace_value': np.nan,
                   'fraction_value': 1.0}

    # read in hazard data from netcdf and use the previously defined variables
    # to help define the data
    hazard1 = Hazard.from_netcdf(file_name, **hazard_args)

    hazard1.check()  #This needs to come before the plots

    return hazard1


def read_exposures_csv(exposure_csv):
    """
        read in exposure information from a csv file
        (e.g. location of primary or distribution substations)

        Inputs
        ------
           data: csv with lon and lat

        Returns
        -------
             exposure information in climada format
        """

    assets = pd.read_csv(exposure_csv)

    # drop missing lines
    assets= assets.dropna(how="all")

    # set the value of the asset to be 1. This represents 1 day of over heating.
    assets["value"] = 1

    # subset the data we want and set appropriate variable names
    data = assets[['latitude','longitude', 'value']]

    # crreat exposure class
    exp = Exposures(data)
    return exp


def read_exposures_nc(exposure_netcdf, var_name, ssp_year):
    """
        read in exposure information from a netCDF file
        (e.g. gridded population data)

        Inputs
        ------
           data: gridded exposure data

        Returns
        -------
             exposure information in climada format
        """

    exp = Exposures.from_netcdf(exposure_netcdf, str(var_name), int(ssp_year))

    return exp


def exposure_instance(exp, exp_unit):
    """
    Create an exposure instance of the Exposures class for Heat stress and
    produce some plots

    Inputs
    ------
       exposure instance created using CLIMADAs Exposures function


    Returns
    -------
         exp: (Object) exposure instance for an exposure class
    """


    # set geometry attribute (shapely Points) from GeoDataFrame from
    # latitude and longitude
    exp.set_geometry_points()

    # always apply the check() method in the end. It puts in metadata that has
    # not been assigned, and causes an error if missing mandatory data
    exp.check()

    #Set the value unit
    exp.value_unit = exp_unit

    return exp


def set_entity(exp_instance):
    """
    Put the exposures into the Entity class

    Inputs
    ------
       exp_instance: (Object) exposure instance for an exposure class

    Returns
    -------
         ent: (Object)  an Entity class with an instance of exposures
    """
    ent = Entity()
    ent.exposures = exp_instance
    return ent


def define_impact_fn(p1, p2, haz_type, int_unit, _id=1):
    """
    Adds impact functions to Entity class

    Inputs
    ------
       p1, p2: parameters relevant for vulnerability equation
       haz_type: e.g. 'Heatstress'
       int_unit: e.g. degC
       _id=1

    Returns
    -------
         imp_fun: impact funtcion instances for given hazard type
    """

    def imp_arc1(hum, p1, p2):
        return 1-1/(1+(p1/hum)**(p2))

    imp_fun = ImpactFunc()
    imp_fun.haz_type = haz_type
    imp_fun.id = _id
    imp_fun.intensity_unit =  int_unit
    imp_fun.intensity = np.linspace(0, 100, num=1000000)
    imp_fun.mdd = np.array([imp_arc1(hum, p1, p2) for hum in imp_fun.intensity])
    imp_fun.paa = np.repeat(1, len(imp_fun.intensity))
    imp_fun.check()

    return imp_fun


def calc_impact(ent1, hazard1):
    """
    Create an impact class

    Inputs
    ------
       ent1: (Object)  an Entity class with a Heat stress instance of
         exposures and impact function instance for Heat stress

    Returns
    -------
         imp1: (Object) an Object that contains the results of impact
         calculations
    """

    imp1 = Impact()

    imp1.calc(ent1.exposures, ent1.impact_funcs, hazard1, save_mat='True')
    print(np.shape(imp1.imp_mat.toarray()))
    return imp1


def read_hazard(warming_level, ens_mem):
    """
    Read in hazard data

    Inputs
    ------
       warming_level: current, WL2, WL4
       ens_mem: one of 12 UKCP ensemble members

    Returns
    -------
         netcdf_file_path: path of data used
         netcdf_file: Dataset containing hazard data
    """
    if warming_level == 'current':
        netcdf_file_path = glob.glob(
                DATA_DIR + f'/UKCP_BC/Timeseries_{ens_mem}_humidex_1998*')
    else:
        netcdf_file_path = glob.glob(
            DATA_DIR + f'/UKCP_BC/Timeseries_{ens_mem}_humidex*{warming_level}*')

    print(netcdf_file_path)

    # load in the hazard data (mean_temperature)

    netcdf_file = Dataset(netcdf_file_path[0])
    return netcdf_file_path[0], netcdf_file


def check_years_ge_15(nc_data):

    nyears = round(nc_data.dimensions['time'].size/360)

    return nyears >= 15


def climada_plots(hazard,impf_set,exp,imp):
    ##### Internal CLIMADA plotting functionality #####
    fig = plt.figure(figsize=(10,10))
    gs = GridSpec(nrows=2, ncols=2)
    ax1 = fig.add_subplot(gs[0, 0],projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0],projection=ccrs.PlateCarree())
    ax4 = fig.add_subplot(gs[1, 1],projection=ccrs.PlateCarree())

    # Hazard plot for an individual event
    hazard.plot_intensity(axis=ax1, event=4518, vmin=0, vmax=110)
    ax1.title.set_text('Humidex Intensity for individual event')
    # Vulnerability function
    impf_set.plot(axis=ax2)
    ax2.title.set_text('Vulnerability function')
    ax2.set_xlabel('Humidex Intensity ($^\circ$C)')
    # Exposure dataset
    norm = colors.LogNorm(vmin=1.0e1, vmax=1.0e6)
    exp.plot_scatter(pop_name=False, axis=ax3, norm=norm,s=11)
    ax3.title.set_text('Exposure')
    # Impact
    # calc impact for largest 'event'
    impact_at_events_exp = imp._build_exp_event(4518)
    impact_at_events_exp.plot_scatter(axis=ax4,pop_name=False, norm=norm,s=16)
    ax4.title.set_text('Impact')

    plt.tight_layout()
    plt.savefig(
          OUT_DIR + 'climada_plotting.png', dpi=500)
    plt.close()#plt.show(block=True)


def main():

    # define hazard parameters
    variable = 'humidex'
    warming_level = 'WL4' # can't find the data for WL2

    # define exposure parameters
    ssp = 2
    ssp_year = 2041
    exposure_path = DATA_DIR + '/UKSSPs/Employment_SSP'+str(ssp)+'_12km_Physical.nc'
    exp_nc = True
    exp_unit = 'Days'
    exp_var = 'employment'

    # define vulnerability function parameters
    p1 = 54.5
    p2 = -4.1
    haz_type = 'Heatstress'
    int_unit = 'degC'

    # run framework for all ensemble members
    ens_mem = ['01']#, '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15']

    for i_ens_mem in ens_mem:

        nc_path, nc = read_hazard(warming_level=warming_level, ens_mem=i_ens_mem)

        if check_years_ge_15(nc): # check hazard data contains at least 15 days

            print("\n**** Ensemble number:", i_ens_mem)

            # read in hazard data
            print("\n *** Reading in the hazard data \n")
            hazard = define_hazard(nc_path,nc,variable,haz_type,custom_nyears=True)

            # read in exposure data and create exposure instance
            print("\n *** Reading in the exposure data \n")
            if exp_nc:
                exp = read_exposures_nc(exposure_path, exp_var, ssp_year)
            else:
                exp = read_exposure_csv(exposure_path)
            exp_inst = exposure_instance(exp, exp_unit)
            exp_inst.ref_year = ssp_year

            # add exposure instance to your entity
            print(" *** Putting the exposures into the entity class \n")
            ent = set_entity(exp_inst)

            # add your impact function to the impact function set
            print(" *** Reading in the impact function \n" )
            impf_set = ImpactFuncSet()
            imp_fun = define_impact_fn(p1, p2, haz_type, int_unit)
            impf_set.append(imp_fun)
            ent.impact_funcs = impf_set

            # calculate impact
            print(" *** Calculating the impact \n")
            imp = calc_impact(ent, hazard)

            imp.write_csv(OUT_DIR + "/ens_" + i_ens_mem.zfill(2) +
                            "_" + str(warming_level) + ".csv")
            print (OUT_DIR + "/ens_" + i_ens_mem.zfill(2) +
                            "_" + str(warming_level) + ".csv")

            # Use plotting functionality
            climada_plots(hazard,impf_set,exp,imp)

        else:
            print(f"\n\nEnsemble member {i_ens_mem} at warming level = {warming_level} has less than 15 years of data")

if __name__ == "__main__":
    main()
