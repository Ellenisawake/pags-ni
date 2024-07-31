import os
import cv2
import json
import numpy as np
from image_utils import longi_lati_to_pixel_coords
from data_utils import ROOT_DIR, get_coordinates_from_json



'''
Steps of producing the OSM input
GeoJSON preparation:
    Download OSM chunk file from https://download.geofabrik.de/ (choose continent then country)
        Example downloaded file: ireland-and-northern-ireland-latest.osm
    Use tools such as osmosis to extract features inside a city bounding box, corresponding to satellite corner coordinates
        https://wiki.openstreetmap.org/wiki/Osmosis
        https://wiki.openstreetmap.org/wiki/Osmosis/Polygon_Filter_File_Format
        Input required: city boundary file, prepared using satellite corner coordinates
        Example extracted file: belfast_extracted.osm
    Use tools such as osmtogeojson to convert the OSM file to GeoJSON which can then be processed by the below functions
        https://tyrasd.github.io/osmtogeojson/
        If the conversion fails with heap memory error it could be because there are too many values in the osm file.
        If this happens use osmosis to further split the osm into several smaller chunks, then convert them one by one.
        Put the converted geojson files into a 'geojson_files' subfolder under the root
OSM feature tag selection:
    Consolidate a list green/blue space tags in dictionary data type as below GREEN_CLASSES
Running the script:
    Use a function depending on single or multiple split geojson files to produce the OSM input images.
    generated osm input are categorical (one image for one tag)
    generated images saved in a subfolder named 'geojson_files'
'''

''' Tthe set of greenspace tags that we use as input to the machine learning model.
A similar dictionary like this need to be collected to create blue space input '''
GREEN_CLASSES = {
    'tourism': ['camp_site', 'picnic_site'],
    'landuse': ['village_green', 'recreation_ground', 'meadow', 'grass', 'greenfield', 'forest'],
    'leisure': ['park', 'nature_reserve', 'garden', 'dog_park', 'common'],
    'natural': ['wood', 'moor', 'heath', 'grassland', 'fell', 'scrub', 'tree', 'tree_row', 'wetland'],
    'route': ['hiking', 'foot'],
    'building': ['greenhouse'],
    'highway': ['footway'],
}  # 7 land types, 26 specific tags
# Numbered from 1 (order does not matter)
GREENSPACE_TAGS = {'camp_site': 1, 'picnic_site': 2, 'village_green': 3, 'recreation_ground': 4, 'meadow': 5,
                 'grass': 6, 'greenfield': 7, 'forest': 8, 'park': 9, 'nature_reserve': 10, 'garden': 11,
                 'dog_park': 12, 'common': 13, 'wood': 14, 'moor': 15, 'heath': 16, 'grassland': 17, 'fell': 18,
                 'scrub': 19, 'tree': 20, 'tree_row': 21, 'wetland': 22, 'hiking': 23, 'foot': 24, 'greenhouse': 25,
                 'footway': 26}


def draw_osm_polygon_line_features(features, greenspace_classes, mask, cat, city_coords, sat_w, sat_h):
    # assume single channel mask
    num_entries = 0
    for con in features:
        if con['type'] == 'Feature':
            properties = con['properties']
            # check if type match the category
            type = None
            for sc in greenspace_classes.keys():
                if sc in properties.keys():
                    if properties[sc] in greenspace_classes[sc]:
                        type = properties[sc]
            if type != cat:
                continue
            num_entries += 1
            geometry = con['geometry']
            if geometry['type'] == 'Polygon':
                for poly in geometry['coordinates']:
                    points = np.asarray(poly)
                    pixel_coords = longi_lati_to_pixel_coords(points[:, 0], points[:, 1],
                                                                          city_coords[0],
                                                                          city_coords[1],
                                                                          city_coords[2],
                                                                          city_coords[3], sat_w, sat_h)
                    pixel_coords = pixel_coords.astype(np.int32)
                    cv2.fillPoly(mask, pts=[pixel_coords], color=255)
            elif geometry['type'] == 'MultiPolygon':
                for poly in geometry['coordinates']:
                    for p in poly:
                        points = np.asarray(p)
                        pixel_coords = longi_lati_to_pixel_coords(points[:, 0], points[:, 1],
                                                                              city_coords[0],
                                                                              city_coords[1],
                                                                              city_coords[2],
                                                                              city_coords[3], sat_w, sat_h)
                        pixel_coords = pixel_coords.astype(np.int32)
                        cv2.fillPoly(mask, pts=[pixel_coords], color=255)
            elif geometry['type'] == 'LineString':
                points = np.asarray(geometry['coordinates'])
                pixel_coords = longi_lati_to_pixel_coords(points[:, 0], points[:, 1],
                                                                      city_coords[0],
                                                                      city_coords[1],
                                                                      city_coords[2],
                                                                      city_coords[3], sat_w, sat_h)
                for i in range(len(points) - 1):
                    pt1, pt2 = pixel_coords[i], pixel_coords[i + 1]
                    cv2.line(mask, pt1, pt2, color=255, thickness=2)
            elif geometry['type'] == 'MultiLineString':
                lines = np.asarray(geometry['coordinates'], dtype=object)
                for line in lines:
                    points = np.asarray(line)
                    pixel_coords = longi_lati_to_pixel_coords(points[:, 0], points[:, 1],
                                                                          city_coords[0],
                                                                          city_coords[1],
                                                                          city_coords[2],
                                                                          city_coords[3], sat_w, sat_h)
                    for i in range(len(points) - 1):
                        pt1, pt2 = pixel_coords[i], pixel_coords[i + 1]
                        cv2.line(mask, pt1, pt2, color=255, thickness=2)
    return mask, num_entries


# Use this function for cities with a single geojson
def generate_us_osm_green_images(tags, classes, space_type='greenspace'):
    image_name = 'osm_nogroup'
    # naming: city_date, date is the capturing date of the planetscope satellite data
    cities = ['losangeles_230420', 'newyork_230601', 'seattle_230702', ]
    size_json_file = os.path.join(ROOT_DIR, f'us_satellite_size.json')
    coords_dir = os.path.join(ROOT_DIR, 'city_coordinates')
    geojson_dir = os.path.join(ROOT_DIR, 'geojson_files')
    mask_cat_save_dir = os.path.join(ROOT_DIR, f'osm_{space_type}')
    os.makedirs(mask_cat_save_dir, exist_ok=True)
    f = open(size_json_file)
    satellite_sizes = json.load(f)
    f.close()
    for city_date in cities:
        city, date = city_date.split('_')
        if city not in satellite_sizes.keys():
            print(f'Skipping {city} because satellite size not given in {size_json_file}!')
            continue
        print(f'processing {city}_{date}')
        osm_json_file = os.path.join(geojson_dir, f'{city}_extracted.geojson')
        if not os.path.exists(osm_json_file):
            print(f'osm_json_file not found: {osm_json_file}')
            continue
        coords_json_file = os.path.join(coords_dir, f'{city}_coordinates.json')
        if not os.path.exists(coords_json_file):
            print(f'coords_json file not found: {coords_json_file}')
            continue
        city_coords = get_coordinates_from_json(coords_json_file)
        sat_h, sat_w = satellite_sizes[city]['height'], satellite_sizes[city]['width']
        with open(osm_json_file) as f:
            contents = json.load(f)
            contents = contents['features']
            for cat in tags.keys():
                # num_entries = 0
                mask = np.zeros((sat_h, sat_w), dtype=np.uint8)
                mask_img_to_save = os.path.join(mask_cat_save_dir, f'{city}_{image_name}{tags[cat]}.png')
                mask, num_entries = draw_osm_polygon_line_features(contents, classes, mask, cat, city_coords, sat_w, sat_h)

                cv2.imwrite(mask_img_to_save, mask)
                print(f'{city}-{cat}: {num_entries} entries plotted!')


''' This function is used to create OSM input when the geojson file for one city is split into several files.
 This is required when the city is huge (many US cities) and osm to geojson conversion fails due to java heap memory error.
 It runs on one city only, so the city & date needs to be used manually selected if more than one city. 
 Required input: GeoJSON files for the city, named as city{#}_extracted.geojson. '''
def generate_us_osm_green_images_split_geojson(tags, classes, space_type='greenspace'):
    # number of geojson file splits after osm to geojson conversion
    # different number for different cities, require manual input
    num_splits = 4
    image_name = 'osm_nogroup'
    city, date = 'losangeles', '230420'  # 'newyork', '230601'
    size_json_file = os.path.join(ROOT_DIR, f'us_satellite_size.json')
    geojson_dir = os.path.join(ROOT_DIR, 'geojson_files')
    mask_cat_save_dir = os.path.join(ROOT_DIR, f'osm_{space_type}')
    os.makedirs(mask_cat_save_dir, exist_ok=True)
    coords_json_file = os.path.join(ROOT_DIR, f'city_coordinates/{city}_coordinates.json')
    city_coords = get_coordinates_from_json(coords_json_file)
    f = open(size_json_file)
    satellite_sizes = json.load(f)
    f.close()
    sat_h, sat_w = satellite_sizes[city]['height'], satellite_sizes[city]['width']
    if city not in satellite_sizes.keys():
        print(f'Skipping {city} because satellite size not given in {size_json_file}!')
        exit(0)
    for cat in tags.keys():
        mask = np.zeros((sat_h, sat_w), dtype=np.uint8)
        mask_img_to_save = os.path.join(mask_cat_save_dir, f'{city}_{image_name}{tags[cat]}.png')
        total_entry = 0
        for n in range(1, num_splits+1):
            osm_json_file = os.path.join(geojson_dir, f'{city}{n}_extracted.geojson')
            if not os.path.exists(osm_json_file):
                print(f'osm_json_file not found: {osm_json_file}')
                continue
            with open(osm_json_file) as f:
                contents = json.load(f)
                contents = contents['features']
                mask, num_entries = draw_osm_polygon_line_features(contents, classes, mask, cat, city_coords, sat_w, sat_h)
                total_entry += num_entries
        cv2.imwrite(mask_img_to_save, mask)
        print(f'{city}-{cat}: {total_entry} entries plotted!')


if __name__ == "__main__":
    ''' Change the tags and class lists following same format as green space to produce blue space images.
        Change space_type accordingly to save output into different folders. '''
    # Use this function for cities with a single geojson
    generate_us_osm_green_images(GREENSPACE_TAGS, GREEN_CLASSES, space_type='greenspace')

    # Use this function for cities multiple geojson files, go into the function and make changes accordingly
    # generate_us_osm_green_images_split_geojson(GREENSPACE_TAGS, GREEN_CLASSES, space_type='greenspace')