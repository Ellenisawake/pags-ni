import os
import numpy as np
import cv2
import json
import zipfile
import rasterio
from rasterio.plot import reshape_as_image
import image_utils


from data_utils import ROOT_DIR


'''
Steps of producing the satellite input:
PlanetScope data downloading 
    From https://www.planet.com/explorer/
    GeoTIFF, surface reflectance - 4 band (RGB+NIR)
    Choose "Clip Items to AOI", "Composite items" and "Harmonize"
    Downloaded example file: newyork-230601-4bands_psscene_analytic_sr_udm2.zip
    Place the downloaded zip files into a subfolder under root names 'PlanetScope'
Unzip the downloaded file 
    Function: unzip_planet_us_data()
    One zip into one folder such as newyork_230601
Process the unzipped composite.tiff (several temporary file will be generated too)
    CRS reprojection: reproject_and_calculate_ndvi()
    NDVI calculation: reproject_and_calculate_ndvi()
    Extract RGB 3 band image from the 4-band image (using GDAL)
        https://gdal.org/index.html
    Calibrate raw image value range to 8 bit color image
        Function: extract_and_calibrate_bands()
        RGB channels are calibrated from the individual 3 band image statistics
        Function: calibrate_nir_band()
        NIR channel is calibrated using statistics from all the cities
'''



def get_min_max_val_from_file(log_file):
    with open(log_file) as f:
        lines = f.readlines()
    low = min([float(e.split('=')[-1].split(',')[0]) for e in lines])
    high = max([float(e.split('=')[-1].split(',')[1]) for e in lines])
    return low, high


def unzip_planet_us_data():
    # unzipped files will be placed into further subfolders under 'PlanetScope'
    data_dir = os.path.join(ROOT_DIR, 'PlanetScope')
    # example zip files, change to any other downloaded zip to unzip
    zips = ['newyork-230601-4bands_psscene_analytic_sr_udm2.zip', ]
    # unzip downloaded zip
    for zip_file in zips:
        os.chdir(data_dir)
        base_name = os.path.basename(zip_file)
        city = base_name.split('-')[0]
        date = base_name.split('-')[1]
        unzip_folder = os.path.join(data_dir, f'{city}_{date}')
        if os.path.exists(unzip_folder):
            print(f'Skipping {city}_{date}')
            continue
        print(f'Extracting zip {city}_{date}')
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(unzip_folder)
        print(f'City {city}_{date} unzipped to {unzip_folder}')


def reproject_and_calculate_ndvi():
    # subfolders under 'PlanetScope', where each stores unzipped files for one city
    cities = ['losangeles_230420', 'newyork_230601', 'seattle_230702', ]
    # https://developers.planet.com/docs/apis/data/sensors/
    # 4band order : 1-blue, 2-green, 3-red, 4-NIR
    chn_red, chn_green, chn_blue, chn_nir = 3, 2, 1, 4
    data_dir = os.path.join(ROOT_DIR, 'PlanetScope')
    img_raw = 'composite.tif'  # original 4band
    img_reproj = 'composite_reproj.tif'  # temporary file, after reprojection and boundary removal
    img_ndvi = 'composite_ndvi.tif'  # output NDVI file

    for city_date in cities:
        city, date = city_date.split('_')
        print(f'processing {city}_{date}')
        working_dir = os.path.join(data_dir, f'{city}_{date}')
        os.chdir(working_dir)
        # order: re-project, ndvi, re-assemble, re-color
        # re-projection, boundary removal
        # https://gis.stackexchange.com/questions/48949/epsg-3857-or-4326-for-web-mapping
        # actual coordinates should be in 4326, while images should be 3857
        image_utils.reproject_satellite_image_crs_to_epsg(img_raw, img_reproj, dst_crs='EPSG:3857')

        # calculate ndvi
        current_folder = os.getcwd()  # Get the current folder
        xml_file = None
        for file in os.listdir(current_folder):
            if file.endswith(".xml"):
                xml_file = file
        if xml_file:
            image_utils.calculate_ndvi_planetscope(img_reproj, xml_file, img_ndvi, chn_red, chn_nir)
            print(f'raw NDVI file written at {img_ndvi}')
        else:
            print(f'**** Skipping NDVI for {city}_{date}: xml file not found ****')


def extract_and_calibrate_bands():
    hist_thresh = 0.99  # thresholding percentage to filter out outlier pixels
    cities = ['losangeles_230420', 'newyork_230601', 'seattle_230702', ]
    data_dir = os.path.join(ROOT_DIR, 'PlanetScope')
    chn_red, chn_green, chn_blue, chn_nir = 3, 2, 1, 4
    img_reproj = 'composite_reproj.tif'  # temporary file, after reprojection and boundary removal
    img_rgb = 'composite_rgb.tif'  # temporary file, reordered 3band RGB
    img_rgb_clipped = 'composite_rgb_clipped.tif'
    img_rgb_colored = 'composite_final.tif'  # temporary file, 3band RGB color stretched
    gdal_log_file = 'gdalinfo_log.txt'  # temporary file, logged min and max value statistics etc
    for city_date in cities:
        city, date = city_date.split('_')
        print(f'processing {city}_{date}')
        working_dir = os.path.join(data_dir, f'{city}_{date}')
        os.chdir(working_dir)
        # re-assemble
        command = f'gdal_translate {img_reproj} {img_rgb} -b {chn_red} -b {chn_green} -b {chn_blue} -co COMPRESS=DEFLATE -co PHOTOMETRIC=RGB'
        os.system(command)
        # re-color
        command = f'gdalinfo -mm {img_rgb} | grep Min/Max > {gdal_log_file}'
        os.system(command)
        min_val, max_val = get_min_max_val_from_file(gdal_log_file)
        src = rasterio.open(img_rgb)
        img_data = src.read()
        # histogram filtering with a given threshold
        num_bins = int(max_val - min_val) // 2
        counts, bin_edges = np.histogram(img_data, bins=num_bins)
        cs = np.cumsum(counts) / np.sum(counts)
        threshold = np.argwhere(cs > hist_thresh)[0]
        clip_val = bin_edges[threshold[0]]
        print(f'min_val:{min_val} max_val:{max_val} threshold:{clip_val}')
        # clip the raw values first using threshold
        img_data = np.clip(img_data, min_val, clip_val)
        kwargs = src.meta
        src.close()
        with rasterio.open(img_rgb_clipped, 'w', **kwargs) as dst:
            dst.write(img_data)
        # then stretch the raw values to 16bit image
        command = f'gdal_translate {img_rgb_clipped} {img_rgb_colored} -scale {min_val} {clip_val} 0 65535 -exponent 0.5 -co COMPRESS=DEFLATE -co PHOTOMETRIC=RGB'
        os.system(command)

        # rescale for NDVI
        img_ndvi = os.path.join(working_dir, 'composite_ndvi.tif')
        img_final = os.path.join(working_dir, 'ndvi255.png')  # final output 8bit NDVI image
        src_ndvi = rasterio.open(img_ndvi)
        band_ndvi = src_ndvi.read()
        clip_min, clip_max = 0.0, 1.0
        img_data = np.clip(band_ndvi, clip_min, clip_max)
        img_data *= 255
        img_data = img_data.astype(np.uint8)
        img_data = reshape_as_image(img_data)
        cv2.imwrite(img_final, img_data)
        src_ndvi.close()

        # rescale for red
        img_rgb = os.path.join(working_dir, 'composite_final.tif')  # extract 3band image, in RGB order
        img_red = os.path.join(working_dir, 'red255.png')
        src = rasterio.open(img_rgb)
        band_red = src.read(1)  # first channel in reorganised 3band (RGB)
        img_data = np.expand_dims(band_red, axis=-1)
        cv2.imwrite(img_red, img_data)  # final output 8bit red image
        # rescale for green
        img_green = os.path.join(working_dir, 'green255.png')
        band_green = src.read(2)  # second channel in reorganised 3band (RGB)
        img_data = np.expand_dims(band_green, axis=-1)
        cv2.imwrite(img_green, img_data)  # final output 8bit green image
        # rescale for blue
        img_blue = os.path.join(working_dir, 'blue255.png')
        band_blue = src.read(3)  # third channel in reorganised 3band (RGB)
        img_data = np.expand_dims(band_blue, axis=-1)
        cv2.imwrite(img_blue, img_data)  # final output 8bit blue image
        src.close()
        print(f'Finish processing {city}_{date} ^^')


def calibrate_nir_band():
    hist_thresh = 0.99  # thresholding percentage to filter out outlier pixels
    cities = ['losangeles_230420', 'newyork_230601', 'seattle_230702', ]
    data_dir = os.path.join(ROOT_DIR, '../PlanetScope/USA')
    chn_nir = 4
    us_nir_clip_max = 0.0
    # calculate the clip value using NIR channel from all cities
    for city_date in cities:
        city, date = city_date.split('_')
        print(f'processing {city}_{date}')
        working_dir = os.path.join(data_dir, f'{city}_{date}')
        os.chdir(working_dir)
        img_reproj = os.path.join(working_dir, 'composite_reproj.tif')  # 0, 65535
        src = rasterio.open(img_reproj)
        band_nir = src.read(chn_nir)  # chn_nir=4 in original 4band
        counts, bin_edges = np.histogram(band_nir, bins=500)
        cs = np.cumsum(counts) / np.sum(counts)
        threshold = np.argwhere(cs > hist_thresh)[0]
        clip_val = bin_edges[threshold[0]]
        print(f'{city} {date} min:{np.min(band_nir)} max:{np.max(band_nir)} mean:{np.mean(band_nir)} clip_val:{clip_val}')
        us_nir_clip_max = max(us_nir_clip_max, clip_val)

    # rescale for NIR using the calculated clip value
    for city_date in cities:
        city, date = city_date.split('_')
        print(f'processing {city}_{date}')
        working_dir = os.path.join(data_dir, f'{city}_{date}')
        os.chdir(working_dir)
        # rescale for NIR
        img_reproj = os.path.join(working_dir, 'composite_reproj.tif')  # 0, 65535
        src = rasterio.open(img_reproj)
        img_nir = os.path.join(working_dir, 'nir255.png')
        band_nir = src.read(chn_nir)  # chn_nir=4 in original 4band
        clip_min, clip_max = 0.0, us_nir_clip_max
        img_data = np.clip(band_nir, clip_min, clip_max)
        img_data = img_data.astype(np.uint8)
        img_data = np.expand_dims(img_data, axis=-1)
        cv2.imwrite(img_nir, img_data)
        src.close()
        print(f'Finish processing {city}_{date} ^^')


if __name__ == "__main__":
    # runs in order, next function use output from previous ones
    # divided into separate functions so that if one step runs into problem, just that step needs a re-run
    unzip_planet_us_data()
    reproject_and_calculate_ndvi()
    extract_and_calibrate_bands()
    calibrate_nir_band()