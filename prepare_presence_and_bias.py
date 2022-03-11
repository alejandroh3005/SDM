"""
Author: Alejandro Hernandez
Last edited: Mar 11, 2022

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
import arcgisscripting

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
crop = "almond"
crop_id = "75"     # Refer to crop_dictionary for codes
crop_dir = r"D:\sdm_data\california\crop\ca_{}_{}".format(crop, crop_id)
geo_db_dir = r"C:\Users\alega\PyScriptTutorial\Data\specialty.gdb"
wgs_coordinate_system = r"{0}\Data\14.prj".format(crop_dir)     # https://spatialreference.org/ref/sr-org/14/

print "Creating point folder for {}: ".format(crop),  # end comma makes end of print statement a space, not newline
point_dir = r"{}\points".format(crop_dir)
try:
    os.mkdir(point_dir)
    print "OK"
except WindowsError:
    print "already exists.".format()

year_index = 0
for raster_name in os.listdir(geo_db_dir):
    if raster_name == "c{}_{}.asc".format(crop_id, year_index):
        try:
            raster = geo_db_dir + "\\" + raster_name
            print "Processing {} from {}: ".format(crop, year_index),
            # convert to points
            crop_points = r"{}\{}_{}".format(point_dir, crop, year_index)
            arcpy.RasterToPoint_conversion(in_raster=raster,
                                           out_point_features=crop_points)
            print "converted, ",
            # project points
            arcpy.Project_management(in_dataset=crop_points,
                                     out_dataset=crop_points,
                                     out_coor_system=wgs_coordinate_system)
            print "projected, ",
            # add XY coords
            arcpy.AddXY_management(in_features=crop_points)
            print "and spatial-ized."
        except arcgisscripting.ExecuteError as err:
            print "FAILED ({}).".format(err.message)
        year_index += 1

exit(0)  # Exit with Error Code 0 (no errors)

# Rarefy projected points
# Convert rarefied point SHAPE files to CSV files.


# Create bias file from rarefied points (cannot automate with ArcPy)
bias_file = r"C:\Users\alega\PycharmProjects\SDM\data\madagascar\bias\BiasFile\Furcifer_oustaleti.asc"
# IGNORE: bias_points = r"C:\Users\alega\PycharmProjects\SDM\data\madagascar\bias\Furcifer_oustaleti_points.shp"

bias_dir = r"bias\BiasFile"
# convert to points
bias_points = bias_dir + r"\\" + "{}_points.shp".format(crop)
arcpy.RasterToPoint_conversion(in_raster=bias_file,
                               out_point_features=bias_points)
# add XY coords
arcpy.AddXY_management(in_features=bias_points)