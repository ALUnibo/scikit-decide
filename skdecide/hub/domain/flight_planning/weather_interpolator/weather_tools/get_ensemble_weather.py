from datetime import datetime
from time import time, sleep

import numpy as np
import os
import sys
import logging
import random
import urllib.request as request

import cfgrib
from math import floor, ceil, sqrt, atan2

from skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.parser_pygrib import \
    GribPygribUniqueForecast
from skdecide.utils import get_data_home

logger = logging.getLogger()


def get_absolute_path(filename, relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(filename), relative_path))


def get_latest_gefs(files_id=None):
    """
    Get the today's GEFS data
    :param files_id: List of files to download, int(s) multiples of 3
    :return: List of files
    """
    if files_id is None:
        files_id = [0]
    current_date = str(datetime.now().date()).replace("-", "")

    exportdir = get_absolute_path(
        __file__,
        f"{get_data_home()}/weather/gefs/" + current_date
    )
    if not os.path.exists(exportdir):
        os.makedirs(exportdir)
    list_files = [os.path.join(exportdir, x) for x in os.listdir(exportdir) if "idx" not in x]
    for i in files_id:
        filename = "geavg.t00z.pgrb2a.0p50.f00" + str(i)
        filepath = os.path.join(exportdir, filename)
        if filepath not in list_files:
            logger.info("Downloading GEFS data")
            url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/gefs."
            url += current_date + "/00/atmos/pgrb2ap5/" + filename
            request.urlretrieve(url, filepath)
            request.urlretrieve(url + ".idx", filepath + ".idx")
            list_files = [os.path.join(exportdir, x) for x in os.listdir(exportdir) if "idx" not in x]
        else:
            logger.info("GEFS data already downloaded")
    return list_files


def bilinear_interpolation(x, y, values):
    """
    Bilinear interpolation of a 2D grid of values
    :param x: X coordinate
    :param y: Y coordinate
    :param values: 4 values of the grid
    :return: Interpolated value
    """
    top_left, top_right, bottom_left, bottom_right = values
    top = top_left * (1 - x) + top_right * x
    bottom = bottom_left * (1 - x) + bottom_right * x
    interpolated_value = top * (1 - y) + bottom * y
    return interpolated_value


def get_wind_values(lat: float, lon: float, alt=0, noisy=False, noise_amount=0.15, ds=None) -> tuple:
    """
    Get the wind magnitude and direction at a given latitude, longitude and altitude
    :param lat: Latitude
    :param lon: Longitude
    :param alt: Altitude
    :param noisy: Add noise to the wind values
    :param noise_amount: Amount of noise to add to the wind values
    :param ds: GEFS Wind Dataset
    :return: Wind magnitude and direction
    """

    # ds = xr.open_dataset(test[0], engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
    # ds = cf2cdm.translate_coords(ds, cf2cdm.ECMWF)
    if ds is None:
        test = get_latest_gefs()
        ds = cfgrib.open_datasets(test[0])

    useful_vars = {"tcc": 0, "pwat": 1, "soilw": 2, "u10": 3, "v10": 3, "r2": 4, "t": 5, "u": 6, "v": 6,
                   "w": 7, "gh": 8, "prmsl": 9, "ulwrf": 10, "cape": 11, "cin": 11,
                   "cicep": 12, "crain": 12, "csnow": 12, "dlwrf": 12, "dswrf": 12, "mslhf": 12, "msshf": 12,
                   "sde": 12, "sp": 12, "tp": 12}
    for u_v in useful_vars.keys():
        for idx in range(len(ds)):
            if u_v in ds[idx]:
                useful_vars[u_v] = idx
                break

    s_p = np.array([[ceil(lat), ceil(lon)], [floor(lat), ceil(lon)],
                    [ceil(lat), floor(lon)], [floor(lat), floor(lon)]])
    s_p = np.stack(s_p)
    ds_id_u = useful_vars["u"]
    ds_id_v = useful_vars["v"]
    alt_value = np.abs(ds[ds_id_u].isobaricInhPa.data - alt).argmin()

    wind_dict = {"magnitude": np.sqrt(
        np.square(ds[ds_id_u]["u"][alt_value].data[s_p[:, 0], s_p[:, 1]]) +
        np.square(ds[ds_id_v]["v"][alt_value].data[s_p[:, 0], s_p[:, 1]])),
        "direction": np.arctan2(
            ds[ds_id_v]["v"][alt_value].data[s_p[:, 0], s_p[:, 1]],
            ds[ds_id_u]["u"][alt_value].data[s_p[:, 0], s_p[:, 1]])
    }

    magnitude = bilinear_interpolation(lat - floor(lat), lon - floor(lon), wind_dict["magnitude"])
    direction = bilinear_interpolation(lat - floor(lat), lon - floor(lon), wind_dict["direction"])


    if noisy:
        avg_u_magnitude = np.mean(ds[ds_id_u]["u"][alt_value].data)
        avg_v_magnitude = np.mean(ds[ds_id_v]["v"][alt_value].data)

        avg_magnitude_noise = sqrt(avg_u_magnitude ** 2 + avg_v_magnitude ** 2) * noise_amount
        avg_direction_noise = np.pi * noise_amount

        magnitude += np.random.normal(0, avg_magnitude_noise)
        direction += np.random.normal(0, avg_direction_noise)

    return magnitude, direction, ds


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    """
        List of variables in a 10-12 (isobaricInhPa) x 361 (latitude) x 720 (longitude) grid:
    
        tcc: Total Cloud Cover (lat x lon)
        pwat: Precipitable water (lat x lon)
        soilw: Volumetric soil moisture content (lat x lon)
        u10: 10 metre U wind component (lat x lon)
        v10: 10 metre V wind component (lat x lon)
        r2: 2-metre Relative humidity (lat x lon)
        r: Relative humidity (hPa x lat x lon)
        t: Air temperature (hPa x lat x lon)
        u: U wind component (eastward) (hPa x lat x lon)
        v: V wind component (northward) (hPa x lat x lon)
        w: Lagrangian tendency of air pressure (lat x lon)
        gh: Geopotential height (hPa x lat x lon)
        prmsl: Pressure reduced to Mean Sea Level (lat x lon)
        ulwrf: Upward long-wave radiation flux (lat x lon)
        cape: Convective available potential energy (lat x lon)
        cin: Convective inhibition (lat x lon)
        
        cicep: Categorical Ice Pellets (lat x lon)
        crain: Categorical Rain (lat x lon)
        csnow: Categorical Snow (lat x lon)
        dlwrf: Downward long-wave radiation flux (lat x lon)
        dswrf: Downward short-wave radiation flux (lat x lon)
        mslhf: Mean surface latent heat flux (lat x lon)
        msshf: Mean surface sensible heat flux (lat x lon)
        sde: Snow depth (lat x lon)
        sp: Surface pressure (lat x lon)
        tp: Total precipitation (lat x lon)
    """

    # variable_to_check = "t"
    # ds_id = useful_vars[variable_to_check]
    # if variable_to_check in ds[ds_id]:
    #     print(f"Air Temperature at a pressure of {ds[ds_id].isobaricInhPa[5].data} hectoPascal, "
    #           f"at Latitude {ds[ds_id].latitude[30].data}, longitude {ds[ds_id].longitude[30].data}: "
    #           f"{ds[ds_id]['t'][5][30][30].data} Kelvin")
    # else:
    #     print(f"Variable {variable_to_check} not found in the dataset")

    LAT = 43.60914993286133
    LON = 1.3691602945327759
    ALT = 10

    magnitude, direction, wind_ds = get_wind_values(LAT, LON, ALT, noisy=True, noise_amount=0)

    print(f"Wind magnitude at LAT {LAT}, LON {LON}, ALT {float(ALT)}: {magnitude} m/s, direction: {direction} rad")
