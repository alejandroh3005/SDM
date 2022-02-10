"""
Author: Alejandro Hernandez
Last edited: Jan 20, 2022

STEP 1: Manipulate presence rasters and create bias raster, 2 of our 3 MaxEnt inputs.

The objective of this script is to convert raster data (from species occurrence locations) into points projected on
an appropriate coordinate system, and with a recorded longitude and latitude- all requirements for data as input to
the SDM MaxEnt tool. Simple!

Following this script, the user will want to rarefy the points, then create bias files from them. Unfortunately,
these processes are specific to the ArcMap SDM Toolbox, therefore neither are able to automated with ArcPy and must be
completed "by hand".

Process: Occurrence Raster --> Points --> Projected Points (with Long/Lat) --> Rarefied Points --> Bias File
"""
import os
from datetime import datetime
import arcpy
from arcgisscripting import ExecuteError

crop_dictionary = {69: "Grape",
                   72: "Citrus",
                   75: "Almond",
                   76: "Walnut",
                   204: "Pistachio",
                   # 215: "Avocados",
                   221: "Strawberry",
                   227: "Lettuce",
                   }

# This is unique to each data set
parent_directory = r"C:\Users\alega\PyScriptTutorial"
wgs_coordinate_system = r"{0}\Data\14.prj".format(parent_directory)     # https://spatialreference.org/ref/sr-org/14/
YEARS_OF_DATA = 11
crop = "Almond"
crop_id = "75"     # Refer to crop_dictionary for codes

# Create crop output folder
crop_directory = r"{0}\Outputs\{1}".format(parent_directory, crop)
try: os.mkdir(crop_directory)
except WindowsError as e:
    exit(r"Early termination: Directory '...\{}' already exists.".format(crop))

# Organize crop raster file names
crop_rasters = []
for year in range(YEARS_OF_DATA):
    crop_rasters.append(r"{0}\Data\specialty.gdb\c{1}_{2}".format(parent_directory, crop_id, year))
    break
print datetime.now().strftime("%H:%M - "), "Crop rasters have been organized."

# Convert rasters to points
for year in range(YEARS_OF_DATA):
    try:
        arcpy.RasterToPoint_conversion(in_raster=crop_rasters[year],
                                       out_point_features=r"{0}\{1}_{2}".format(crop_directory, crop, year))
    except ExecuteError as e:
        print "Raster c{0}_{1} could not be converted to points.".format(crop_id, year)
        print e.message
    break
print datetime.now().strftime("%H:%M - "), "Crop rasters have been converted to points.\n"

# Project points to WGS 1984 coordinate system (necessary to rarefy)
for year in range(YEARS_OF_DATA):
    try:
        arcpy.Project_management(in_dataset=r"{0}\{1}_{2}.shp".format(crop_directory, crop, year),
                                 out_dataset=r"{0}\Projected_{1}_{2}.shp".format(crop_directory, crop, year),
                                 out_coor_system=wgs_coordinate_system)
    except ExecuteError as e:
        print "Points c{0}_{1} could not be projected.".format(crop_id, year)
    break
print datetime.now().strftime("%H:%M - "), "Crop points have been projected to WGS 1984 Coordinate System."

# Add XY features to projected points
for year in range(YEARS_OF_DATA):
    try:
        arcpy.AddXY_management(in_features=r"{0}\Projected_{1}_{2}.shp".format(crop_directory, crop, year))
    except ExecuteError as e:
        print "X and Y features for c{0}_{1} could not be added .".format(crop_id, year)
    break
print datetime.now().strftime("%H:%M - "), "Longitude and latitude features have been added to crop points."

# Rarefy projected points

# Convert rarefied point SHAPE files to CSV files.

# Create bias file from rarefied points