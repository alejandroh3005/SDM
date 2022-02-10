from geopandas import read_file, sjoin
import pandas as pd

class SpatialJoin:
    def __init__(self, source):
        PRESENCE_ONLY = True
        N_YEARS = 19
        # Parent directory and file locations
        parent_directory = r"C:\Users\alega\PycharmProjects\SDM"
        if source == "CA":
            rarefied_points = fr"{parent_directory}\CA_Rarefied\Strawberry\strawb_spatially_rarified_locs.shp"
            raster_polygons = [fr"{parent_directory}\CA_Polygons\bio_{num + 1}.shp" for num in range(N_YEARS)]
        elif source == "MD":
            rarefied_points = fr"{parent_directory}\MD_Rarefied\halfoccurence__spatially_rarified_locs.shp"
            raster_polygons = [fr"{parent_directory}\MD_Polygons\bio_{num + 1}.shp" for num in range(N_YEARS)]
        else:
            print("Invalid 'source' parameter.")
            rarefied_points, raster_polygons = None, None

        """Organize Point data into a GeoDataFrame"""
        point_df = read_file(rarefied_points)
        if source == "CA":
            point_df = point_df.rename(columns={"grid_code": "species"})
        elif source == "MD":
            species = point_df["species"].values
            for i in range(len(species)):
                if species[i] == "Furcifer_pardalis": species[i] = 0
                else: species[i] = 1
            point_df["species"] = species
            if PRESENCE_ONLY:  # Remove all FP occurrences
                point_df = point_df[point_df.species != 0]

        """Organize Polygon data into a GeoDataFrame"""
        polygon_df = read_file(raster_polygons[0]).rename(columns={"grid_code": "bio_1"})
        for count, polygon in enumerate(raster_polygons[1:]):
            curr_polygon_df = read_file(polygon).rename(columns={"grid_code": f"bio_{count + 2}"})
            polygon_df = polygon_df.join(curr_polygon_df[f"bio_{count + 2}"])

        """Use Spatial Join tool and save DataFrame to csv file"""
        joined_df = sjoin(point_df, polygon_df, op="within").drop(columns=["geometry", "index_right"])
        self.presence_df = pd.DataFrame(joined_df)  # Convert GeoDataFrame to DataFrame
        if PRESENCE_ONLY:
            self.presence_df.to_csv(f"{source}_presence")
        else:
            self.presence_df.to_csv(f"{source}_presence_absence")


if __name__ == "__main__":
    pass
