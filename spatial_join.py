import os
import pandas as pd
import geopandas as gpd

def create_geodataframes(source="madagascar", species="all", data_dir=r"C:\Users\alega\PycharmProjects\SDM\data"):
    if source not in ["california", "madagascar"]:
        exit("Invalid 'source' parameter.")

    # Set directories and file locations
    rarefied_dir = fr"{data_dir}\{source}\rarefied\{species}"
    bias_dir = fr"{data_dir}\{source}\bias"
    polygon_dir = fr"{data_dir}\{source}\polygons"

    # Organize Point shapefiles into a GeoDataFrame
    point_df = gpd.GeoDataFrame(data=None)
    for point_file in os.listdir(rarefied_dir):
        if point_file.endswith('.shp'):
            point_file = rarefied_dir + "\\" + point_file
            point_df = gpd.read_file(point_file)

    # Organize Bias shapefiles into a GeoDataFrame
    bias_df = gpd.GeoDataFrame(data=None)
    for bias_file in os.listdir(bias_dir):
        if bias_file.endswith('.shp'):
            bias_file = bias_dir + "\\" + bias_file
            bias_df = gpd.read_file(bias_file)
            break

    # Organize Polygon shapefiles into a GeoDataFrame
    var_index = 1
    polygon_df = gpd.GeoDataFrame(data=None)
    for polygon_file in os.listdir(polygon_dir):
        if polygon_file.endswith('.shp'):
            polygon_file = polygon_dir + "\\" + polygon_file
            if var_index == 1:
                polygon_df = gpd.read_file(polygon_file)
                polygon_df = polygon_df[polygon_df['pointid'] > 0].rename(columns={'grid_code': f'bio_{var_index}'})
            else:
                temp_df = gpd.read_file(polygon_file).drop(columns='geometry')
                temp_df = temp_df[temp_df['pointid'] > 0].rename(columns={'grid_code': f'bio_{var_index}'})
                polygon_df = polygon_df.merge(right=temp_df, on='pointid')
            var_index += 1

    """Finalize Occurrence Data Frame"""
    occurrence_joined_df = gpd.sjoin(point_df, polygon_df, op="within")
    pd.DataFrame(occurrence_joined_df).to_csv(path_or_buf=fr"{data_dir}\{source}\occurrence_data.csv")

    """Finalize Background Data Frame"""
    bias_df.crs = "EPSG:4326"  # coordinate reference system
    bias_joined_df = gpd.sjoin(bias_df, polygon_df, op="within")
    pd.DataFrame(bias_joined_df).to_csv(path_or_buf=fr"{data_dir}\{source}\bias_data.csv")

    """Finalize Survey Space Data Frame"""
    pd.DataFrame(polygon_df.drop(columns='geometry')).to_csv(path_or_buf=fr"{data_dir}\{source}\climate_data.csv")


if __name__ == "__main__":
    create_geodataframes()
