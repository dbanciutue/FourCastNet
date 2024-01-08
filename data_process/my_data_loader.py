import h5py
from mpi4py import MPI
import numpy as np
import time
from netCDF4 import Dataset as DS
import os

def process_era5_data(file_paths, dest, channel_start, frmt='nc'):
    channel_idx = channel_start
    for variable_name, source_path in file_paths.items():
        writetofile(source_path, dest, channel_idx, [variable_name], frmt=frmt)
        channel_idx += 1

def writetofile(file_paths, dest):
    batch = 2**4
    rank = MPI.COMM_WORLD.rank
    Nproc = MPI.COMM_WORLD.size

    for variable_name, (src, channel_idx) in file_paths.items():
        print(f"Variable Name: " +  variable_name)
        print(f"Source File: " +  src)
        print(f"channel_idx: " +  src)
        if os.path.isfile(src):
            print("started if")

            rank = MPI.COMM_WORLD.rank
            Nproc = MPI.COMM_WORLD.size
            Nimgtot = 52#src_shape[0]

            Nimg = Nimgtot//Nproc
            base = rank*Nimg
            end = (rank+1)*Nimg if rank<Nproc - 1 else Nimgtot
            idx = base

            fsrc = DS(src, 'r', format="NETCDF4").variables[variable_name]
            fdest = h5py.File(dest, 'a', driver='mpio', comm=MPI.COMM_WORLD)

            start = time.time()
            while idx < end:
                if end - idx < batch:
                    ims = fsrc[idx:end]
                    fdest['fields'][idx:end, channel_idx, :, :] = ims
                    break
                else:
                    ims = fsrc[idx:idx + batch]
                    fdest['fields'][idx:idx + batch, channel_idx, :, :] = ims
                    idx += batch

            ttot = time.time() - start
            hrs = ttot // 3600
            mins = (ttot - 3600 * hrs) // 60
            secs = (ttot - 3600 * hrs - 60 * mins)



###############################################################################################################################################################
dest = "/your/destination/folder/"

#u10 v10 t2m
writetofile('/mnt/qb/goswami/data/era5/single_pressure_level/10m_u_component_of_wind/10m_u_component_of_wind_2018.nc', dest, 0, ['u10'])
writetofile('/mnt/qb/goswami/data/era5/single_pressure_level/10m_v_component_of_wind/10m_v_component_of_wind_2018.nc', dest, 1, ['v10'])
writetofile('/mnt/qb/goswami/data/era5/single_pressure_level/2m_temperature/2m_temperature_2018.nc', dest, 2, ['t2m'])

#sp mslp
writetofile('/mnt/qb/goswami/data/era5/single_pressure_level/surface_pressure/surface_pressure_2018.nc', dest, 3, ['sp'])
writetofile('/mnt/qb/goswami/data/era5/single_pressure_level/mean_sea_level_pressure/mean_sea_level_pressure_2018.nc', dest, 4, ['mslp'])

#t850
writetofile('/mnt/qb/goswami/data/era5/multi_pressure_level/temperature/850/temperature_2018_850.nc', dest, 5, ['t'], 2)

#uvz1000
writetofile('/mnt/qb/goswami/data/era5/multi_pressure_level/u_component_of_wind/1000/u_component_of_wind_2018_1000.nc', dest, 6, ['u'], 3)
writetofile('/mnt/qb/goswami/data/era5/multi_pressure_level/v_component_of_wind/1000/v_component_of_wind_2018_1000.nc', dest, 7, ['v'], 3)
writetofile('/mnt/qb/goswami/data/era5/multi_pressure_level/geopotential/1000/geopotential_2018_1000.nc', dest, 8, ['z'], 3)

#uvz850
writetofile('/mnt/qb/goswami/data/era5/multi_pressure_level/u_component_of_wind/850/u_component_of_wind_2018_850.nc', dest, 9, ['u'], 2)
writetofile('/mnt/qb/goswami/data/era5/multi_pressure_level/v_component_of_wind/850/v_component_of_wind_2018_850.nc', dest, 10, ['v'], 2)
writetofile('/mnt/qb/goswami/data/era5/multi_pressure_level/geopotential/850/geopotential_2018_850.nc', dest, 11, ['z'], 2)

#uvz 500
writetofile('/mnt/qb/goswami/data/era5/multi_pressure_level/u_component_of_wind/500/u_component_of_wind_2018_500.nc', dest, 12, ['u'], 1)
writetofile('/mnt/qb/goswami/data/era5/multi_pressure_level/v_component_of_wind/500/v_component_of_wind_2018_500.nc', dest, 13, ['v'], 1)
writetofile('/mnt/qb/goswami/data/era5/multi_pressure_level/geopotential/500/geopotential_2018_500.nc', dest, 14, ['z'], 1)

#t500
writetofile('/mnt/qb/goswami/data/era5/multi_pressure_level/temperature/500/temperature_2018_500.nc', dest, 15, ['t'], 1)

#z50
writetofile('/mnt/qb/goswami/data/era5/hourly/multi_pressure_level/geopotential/50/geopotential_era5_hourly_2018_50.nc', dest, 16, ['z'], 0)

#r500 
writetofile('/mnt/qb/goswami/data/era5/multi_pressure_level/r_500/r_500_2018.nc', dest, 17, ['r'], 1)

#r850
writetofile('/mnt/qb/goswami/data/era5/multi_pressure_level/r_850/r_850_2018.nc', dest, 18, ['r'], 2)

#tcwv
writetofile('/mnt/qb/goswami/data/era5/single_pressure_level/total_column_water_vapour/total_column_water_vapor_2018.nc', dest, 19, ['tcwv'])
