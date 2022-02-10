"""
Author: Alejandro Hernandez
Last edited: Jan 27, 2022

STEP 2: Manipulate climate rasters and prepare the final part of our 3 MaxEnt inputs.

The objective of this script is to convert bioclimatic raster data into polygons.
These polygons are appropriate input for the Spatial Join tool, which is in a separate script,
because it does not require ArcPy tools (that needs Python 2.7) and allows Python 3.

Process: Bioclimatic Raster --> Points --> Fishnet --> Bioclimatic Polygons
"""
from os import mkdir
from arcpy import RasterToPoint_conversion, CreateFishnet_management, FeatureToPolygon_management

num_of_rasters = 19
origin_coord = '-124.39291513532 32.53550109237899'   # Use format 'X_coord Y_coord'
y_axis_coord = '-124.39291513532 42.53550109237899'
parent_directory = r'C:\Users\alega\PyScriptTutorial'
bioclim_directory = r'{0}\Data\CA_bio'.format(parent_directory)

bio_rasters = [r'{0}\bio_{1}.asc'.format(bioclim_directory, num + 1) for num in range(num_of_rasters)]

mkdir(r'{0}\Points'.format(bioclim_directory))
mkdir(r'{0}\Fishnets'.format(bioclim_directory))
mkdir(r'{0}\Polygons'.format(bioclim_directory))

for num, raster in enumerate(bio_rasters):
    RasterToPoint_conversion(in_raster=raster,
                             out_point_features=r'{0}\Points\bio_{1}.shp'.format(bioclim_directory, num + 1))

    CreateFishnet_management(out_feature_class=r'{0}\Fishnets\bio_{1}.shp'.format(bioclim_directory, num + 1),
                             template=raster, origin_coord=origin_coord, y_axis_coord=y_axis_coord,
                             number_rows=1137, number_columns=1233,
                             geometry_type='POLYLINE')

    FeatureToPolygon_management(in_features=r'{0}\Fishnets\bio_{1}.shp'.format(bioclim_directory, num + 1),
                                out_feature_class=r'{0}\Polygons\bio_{1}.shp'.format(bioclim_directory, num + 1),
                                label_features=r'{0}\Points\bio_{1}.shp'.format(bioclim_directory, num + 1))