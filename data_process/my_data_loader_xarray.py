import os
import xarray as xr
import pandas as pd
import numpy as np

def writetofile(file_paths, dest):
    for variable_name, (src, channel_idx, short_name) in file_paths.items():
        print(f"Variable Name: {variable_name}")
        print(f"Source File: {src}")
        print(f"channel_idx: {channel_idx}")
        print(f"Short Name: {short_name}")
        print(f"________________________________________")

        if os.path.isfile(src):
            # Open the NetCDF file with xarray and Dask
            ds = xr.open_dataset(src, engine='netcdf4', chunks={'time': "auto"})

            # Writing to the destination file using Dask's parallelism
            ds.to_netcdf(dest, mode='a', compute=False, format='netcdf4', engine='netcdf4')


def to_xarray_dataset(file_paths):

    list_ds = []
    for variable_name, (src, channel_idx, short_name) in file_paths.items():
        print(f"Variable Name: {variable_name}")
        print(f"Source File: {src}")
        print(f"channel_idx: {channel_idx}")
        print(f"Short Name: {short_name}")
    
        if os.path.isfile(src):
            ds = xr.open_dataset(src)
            data_variable_names = list(ds.data_vars.keys())
            selected_variable_name = data_variable_names[-1]
            print(f"Detected Data Variable Name: {selected_variable_name}")
            ds = ds.rename_vars({f"{selected_variable_name}":f"{variable_name}"})
            list_ds.append(ds)
        else:
            print(f"File not found: {src}")

        print(f"________________________________________")

    full_ds = xr.merge(objects=list_ds, )
    return full_ds


def my_select(dataset, date, delta_days_past, delta_days_future):
        # Convert the specific date string to a pandas Timestamp
        specific_date_timestamp = pd.to_datetime(date, format='%Y-%m-%d')

        # Create a time range for 15 days before and after the specific date
        time_range = pd.date_range(start=specific_date_timestamp - pd.Timedelta(days=delta_days_past),
                                end=specific_date_timestamp + pd.Timedelta(days=delta_days_future),
                                freq='D')  # 'D' for daily frequency


        # Select the time period in the dataset based on the calculated time range
        return dataset.sel(time=time_range, method="nearest")

def normalize_dataset(dataset, means, stds, file_paths):
    def get_channel_index(variable_name):
        for var_name, (_, channel_idx, _) in file_paths.items():
            if var_name == variable_name:
                return channel_idx
        return None

    for var_name in dataset.data_vars:
         channel_idx = get_channel_index(var_name)
         dataset[var_name] = dataset[var_name] - means[:,channel_idx] / stds[:,channel_idx]

    return dataset

def calculate_relative_humidity(specific_humidity_xarray, year, pressure_level,):    

    # Extract specific humidity and temperature data
    temp_xarray = xr.open_dataset(f"/mnt/qb/goswami/data/era5/multi_pressure_level/temperature/{pressure_level}/temperature_{year}_{pressure_level}.nc")
    temp_xarray = temp_xarray.sel(time = specific_humidity_xarray.time)
    
    specific_humidity = specific_humidity_xarray[f"rh_{pressure_level}"]
    temperature = temp_xarray['t'].values

    # Calculate saturation vapor pressure (e_s) using the temperature data
    e_s = 6.11 * np.exp(17.67 * (temperature - 273.16) / (temperature - 29.65))

    # Calculate saturation mixing ratio (w_s)
    w_s = 0.622 * (e_s / (pressure_level))

    # Calculate relative humidity (r_h)
    relative_humidity = 100 * (specific_humidity / w_s)

    specific_humidity_xarray[f"rh_{pressure_level}"] = relative_humidity

    return specific_humidity_xarray


###############################################################################################################################################################
# Define the year
year = "2018"

# Create a dictionary to store file paths
file_paths = {}

# Surface Level Variables
file_paths['u10'] = (f"/mnt/qb/goswami/data/era5/single_pressure_level/10m_u_component_of_wind/10m_u_component_of_wind_{year}.nc", 0, 'u')
file_paths['v10'] = (f"/mnt/qb/goswami/data/era5/single_pressure_level/10m_v_component_of_wind/10m_v_component_of_wind_{year}.nc", 1, 'v')
file_paths['temp_2m'] = (f"/mnt/qb/goswami/data/era5/single_pressure_level/2m_temperature/2m_temperature_{year}.nc", 2, 't2m')
file_paths['sp'] = (f"/mnt/qb/goswami/data/era5/single_pressure_level/surface_pressure/surface_pressure_{year}.nc", 3, 'sp')
file_paths['mslp'] = (f"/mnt/qb/goswami/data/era5/single_pressure_level/mean_sea_level_pressure/mean_sea_level_pressure_{year}.nc", 4, 'mslp')

# 1000 hPa Level Variables
pressure_1000 = "1000"
file_paths['u_1000'] = (f"/mnt/qb/goswami/data/era5/multi_pressure_level/u_component_of_wind/{pressure_1000}/u_component_of_wind_{year}_{pressure_1000}.nc", 5, 'u')
file_paths['v_1000'] = (f"/mnt/qb/goswami/data/era5/multi_pressure_level/v_component_of_wind/{pressure_1000}/v_component_of_wind_{year}_{pressure_1000}.nc", 6, 'v')
file_paths['z_1000'] = (f"/mnt/qb/goswami/data/era5/multi_pressure_level/geopotential/{pressure_1000}/geopotential_{year}_{pressure_1000}.nc", 7, 'z')

# 850 hPa Level Variables
pressure_850 = "850"
file_paths['t_850'] = (f"/mnt/qb/goswami/data/era5/multi_pressure_level/temperature/{pressure_850}/temperature_{year}_{pressure_850}.nc", 8, 't')
file_paths['u_850'] = (f"/mnt/qb/goswami/data/era5/multi_pressure_level/u_component_of_wind/{pressure_850}/u_component_of_wind_{year}_{pressure_850}.nc", 9, 'u')
file_paths['v_850'] = (f"/mnt/qb/goswami/data/era5/multi_pressure_level/v_component_of_wind/{pressure_850}/v_component_of_wind_{year}_{pressure_850}.nc", 10, 'v')
file_paths['z_850'] = (f"/mnt/qb/goswami/data/era5/multi_pressure_level/geopotential/{pressure_850}/geopotential_{year}_{pressure_850}.nc", 11, 'z')
file_paths['rh_850'] = (f"/mnt/qb/goswami/data/era5/multi_pressure_level/specific_humidity/{pressure_850}/specific_humidity_{year}_{pressure_850}.nc", 12, 'rh')

# 500 hPa Level Variables
pressure_500 = "500"
file_paths['t_500'] = (f"/mnt/qb/goswami/data/era5/multi_pressure_level/temperature/{pressure_500}/temperature_{year}_{pressure_500}.nc", 13, 't')
file_paths['u_500'] = (f"/mnt/qb/goswami/data/era5/multi_pressure_level/u_component_of_wind/{pressure_500}/u_component_of_wind_{year}_{pressure_500}.nc", 14, 'u')
file_paths['v_500'] = (f"/mnt/qb/goswami/data/era5/multi_pressure_level/v_component_of_wind/{pressure_500}/v_component_of_wind_{year}_{pressure_500}.nc", 15, 'v')
file_paths['z_500'] = (f"/mnt/qb/goswami/data/era5/multi_pressure_level/geopotential/{pressure_500}/geopotential_{year}_{pressure_500}.nc", 16, 'z')
file_paths['rh_500'] = (f"/mnt/qb/goswami/data/era5/multi_pressure_level/specific_humidity/{pressure_500}/specific_humidity_{year}_{pressure_500}.nc", 17, 'rh')

# 50 hPa Level Variables
pressure_50 = "50"
file_paths['z_50'] = (f"/mnt/qb/goswami/data/era5/multi_pressure_level/geopotential/{pressure_50}/geopotential_{year}_{pressure_50}.nc", 18, 'z')


# Integrated Total Column Water Vapor
file_paths['tcwv'] = (f"/mnt/qb/goswami/data/era5/single_pressure_level/total_column_water_vapour/total_column_water_vapour_{year}.nc", 19, 'tcwv')

###############################################################################################################################################################

# Define the subset of variables to test
test_vars = {}
test_vars['u10'] = (f"/mnt/qb/goswami/data/era5/single_pressure_level/10m_u_component_of_wind/10m_u_component_of_wind_{year}.nc", 0, "u")
test_vars['v10'] = (f"/mnt/qb/goswami/data/era5/single_pressure_level/10m_v_component_of_wind/10m_v_component_of_wind_{year}.nc", 1, "v")

# Destination file path
#dest = '/mnt/qb/datasets/STAGING/goswami/nvidia_daniel/new_test_output_data_loader_1.h5'
dest = '/mnt/qb/work/goswami/gkd020/output/new_test_output_data_loader_1.h5'


#writetofile(file_paths, dest)

big_ds = to_xarray_dataset(file_paths)
filtered_big_ds = my_select(big_ds, "2018-03-01", 7, 7)
filtered_big_ds = calculate_relative_humidity(filtered_big_ds, 2018, 850)
filtered_big_ds = calculate_relative_humidity(filtered_big_ds, 2018, 500)

means = np.load('/mnt/qb/work/goswami/gkd020/nvidia/github/FourCastNet/additional/stats_v0/global_means.npy')
stds = np.load('/mnt/qb/work/goswami/gkd020/nvidia/github/FourCastNet/additional/stats_v0/global_stds.npy')

normalized_filtered_big_ds = normalize_dataset(filtered_big_ds, means, stds, file_paths)

print("### Memory Usage in GB: ", normalized_filtered_big_ds.nbytes / (1024**3))

print(filtered_big_ds.info)
print(filtered_big_ds.head(5))