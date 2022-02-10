"""
Author: Alejandro Hernandez
Last edited: Feb 9, 2022

Process: World Rasters --> California Rasters
"""
import os
from arcpy import CheckOutExtension, RasterToASCII_conversion
from arcpy.sa import ExtractByMask
x = CheckOutExtension("Spatial")

# set directories to relevant files
data_directory = r"C:/Users/alega/PyScriptTutorial/Data"
world_rasters_directory = r"{}/world_rasters/".format(data_directory)
california_mask = "{}/california_map_bio_30s_01.asc".format(data_directory)

# iterate through world rasters and for each, clip to the exact (polygon) extent of California
n_rasters = 0
for in_raster in os.listdir(world_rasters_directory):
    if in_raster.endswith('.tif'):
        n_rasters += 1
        out_raster = ExtractByMask(in_raster=world_rasters_directory + in_raster, in_mask_data=california_mask)
        out_ascii = "{}/california/ca_bio_30s_{}.asc".format(data_directory, n_rasters)
        RasterToASCII_conversion(in_raster=out_raster, out_ascii_file=out_ascii)
        print("Processed", in_raster)