from sentinelhub import SHConfig
import sentinelhub
import datetime
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Any
import cv2
import json



from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    MosaickingOrder,
    SentinelHubRequest,
)


# bounds for olympic mountains
# lower_left = (-123.826859, 47.509866)
# upper_right = (-123.031568, 47.995574)

# bounds for mt. adams
# upper_right =  [-121.423597, 46.266517]
# lower_left = [-121.593377, 46.134413]

# bounds for goat rocks
# upper_right = (-121.393653, 46.581592)
# lower_left = (-121.534044, 46.441355)

# bounds for north cascades north
# lower_left = (-122.003949, 48.638660)
# upper_right = (-120.562058, 49)

# bounds for north cascades south
# lower_left = (-121.554531, 47.902931)
# upper_right = (-120.656781, 48.638660)

# bounds for alpine lakes
# lower_left = (-121.462225, 47.422051)
# upper_right = (-120.712952, 47.712973)

# bounds for mount rainier area
# lower_left = (-121.902585, 46.776775)
# upper_right = (-121.600930, 46.957635)

# bounds for boulder river area
lower_left = (-121.724466, 48.153833)
upper_right = (-121.645673, 48.227564)


HEIGHT, WIDTH = 1024, 1024

class SentinelTileDownloader:
    def __init__(self, tile_size_km, lon_lower_left, lat_lower_left, lon_upper_right, lat_upper_right, filepath, client_id = None, client_secret = None):
        self.tile_size_km = tile_size_km
        self.lon_lower_left = lon_lower_left
        self.lat_lower_left = lat_lower_left
        self.lon_upper_right = lon_upper_right
        self.lat_upper_right = lat_upper_right
        self.data_folder = filepath

        if not os.path.exists(self.data_folder):
            print(f'file path not found. Creating new folder at {self.data_folder}')
            os.mkdir(self.data_folder)

        # verify passed coordiniates are within the valid ranges for lat and lon
        if not (-180 <= lon_lower_left <= 180 and -90 <= lat_lower_left <= 90):
            raise ValueError('Longitude must be between -180 and 180 degrees')
        if not (-180 <= lon_upper_right <= 180 and -90 <= lat_upper_right <= 90):
            raise ValueError('Latitude must be between -90 and 90 degrees')

        # Check if the lower left coordinates are actually lower and left compared to the upper right
        if lon_lower_left >= lon_upper_right or lat_lower_left >= lat_upper_right:
            raise ValueError("Lower-left coordinates must be less than upper-right coordinates.")

        # if client id is passed, update config
        config = SHConfig()
        if client_id:
            config.sh_client_id = client_id

        # if client secret passed, update config
        if client_secret:
            config.sh_client_secret = client_secret

        config.save()

        message = (
            'Initialized SentinelTileDownloader with: '
            f'lower left = {(lon_lower_left, lat_lower_left)}, '
            f'upper right = {(lon_upper_right, lat_upper_right)}'
        )
        print(message)

    def bbox_square(self, lon, lat, length, precision=8):
        """
        Accepts longitude and latitude of lower left bbox corner
        Accepts length of bbox in km
        Accepts number of decimal places as precision, default to 8
        Returns a bounding box object
        """
        if not -180 <= lon <= 180 or not -90 <= lat <= 90:
            raise ValueError((f"Invalid coordinates: Latitude should be between -90 and 90, and Longitude should be between -180 and 180. Given lat: {lat}, lon: {lon}"))

        # calculate latitude change
        lat_change = length / 111.0

        # Calculate the longitude change, accounting for the latitude
        lon_change = length / (111.0 * math.cos(math.radians(lat)))

        # Calculate the upper right corner coordinates
        upper_right_lon = lon + lon_change
        upper_right_lat = lat + lat_change

        lon = round(lon, precision)
        lat = round(lat, precision)
        upper_right_lon = round(upper_right_lon, precision)
        upper_right_lat = round(upper_right_lat, precision)

        bbox_coordinates = (lon, lat, upper_right_lon, upper_right_lat)
        return BBox(bbox=bbox_coordinates, crs=CRS.WGS84)

    def get_image_square(self, lon, lat):
        """ 
        Accepts longitude in float form
        Accepts latitude in float form
        Accelts side length (in km) in float form
        returns satellite image of box with side length length, and upper left corner of image at the passed coordinates
        """
        config = sentinelhub.SHConfig()
        bbox = self.bbox_square(lon, lat, self.tile_size_km)
        evalscript_true_color = """
        //VERSION=3

        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04"]
                }],
                output: {
                    bands: 3
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02];
        }
        """
        request_true_color = SentinelHubRequest(
            evalscript=evalscript_true_color,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L1C,
                    time_interval=("2023-08-01", "2023-09-15"),
                    mosaicking_order=MosaickingOrder.LEAST_CC,
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
            bbox=bbox,
            size=(HEIGHT, WIDTH),
            config=config,
        )
        true_color_imgs = request_true_color.get_data()
        image = true_color_imgs[0]

        return image

    def canvas_region_rectangular(self):
        """
        Accepts lon and lat for the lower left corner of the desired region.
        Accepts lon and lat for upper right corner of the desired region
        Accepts a tile size for each image (in km)
        Returns a list of bounding box objects (square)
        """

        def get_bbox_row(ll_lon, ll_lat, ur_lon):
            last_bbox = self.bbox_square(ll_lon, ll_lat, self.tile_size_km)
            row = [last_bbox]

            while last_bbox.upper_right[0] < ur_lon:
                last_bbox = row[-1]
                # set bbox lower left lon equal to lon on last bbox right side
                bbox_ll_lon = last_bbox.upper_right[0]

                # keep bbox lower left lat consistent across the whole row
                bbox_ll_lat = last_bbox.lower_left[1]

                # create and append a new bounding box to the row
                new_bbox = self.bbox_square(bbox_ll_lon, bbox_ll_lat, self.tile_size_km)
                row.append(new_bbox)
            
            # add one final bbox to row to ensure complete coverate of the provided coordinates
            last_bbox = row[-1]
            bbox_ll_lon = last_bbox.upper_right[0]
            bbox_ll_lat = last_bbox.lower_left[1]
            new_bbox = self.bbox_square(bbox_ll_lon, bbox_ll_lat, self.tile_size_km)
            row.append(new_bbox)

            return row
        
        first_row = get_bbox_row(self.lon_lower_left, self.lat_lower_left, self.lon_upper_right)
        bbox_grid = [first_row]
        last_lat = self.lat_lower_left


        while last_lat < self.lat_upper_right:

            # for each new row, get the lower left and upper right bbox coordinates for the first element of the prior row
            last_lower_left_row_start = bbox_grid[-1][0].lower_left
            last_upper_right_row_start = bbox_grid[-1][0].upper_right

            # set a new lower left bbox coordinate for a new row using the prior lower left lon and upper right lat
            new_lower_left_lon = last_lower_left_row_start[0]
            new_lower_left_lat = last_upper_right_row_start[1]

            # get the next row of the area and append it to the row
            next_row = get_bbox_row(new_lower_left_lon, new_lower_left_lat, self.lon_upper_right)
            bbox_grid.append(next_row)

            last_lat = new_lower_left_lat

        # add the final row to grid to ensure complete coverate of provided area
        last_lower_left_row_start = bbox_grid[-1][0].lower_left
        last_upper_right_row_start = bbox_grid[-1][0].upper_right
        new_lower_left_lon = last_lower_left_row_start[0]
        new_lower_left_lat = last_upper_right_row_start[1]
        next_row = get_bbox_row(new_lower_left_lon, new_lower_left_lat, self.lon_upper_right)
        bbox_grid.append(next_row)

        return bbox_grid

    def get_images_from_region(self):
        """
        Downloads satellite images for a rectangular region defined by lower-left and upper-right coordinates.

        The region is divided into square tiles of specified size, and images are retrieved for each tile.
        The images are saved as PNG files in the specified output directory.

        Parameters:
        ----------
        region_lon_lower_left : float
            Longitude of the lower-left corner.
        
        region_lat_lower_left : float
            Latitude of the lower-left corner.

        region_lon_upper_right : float
            Longitude of the upper-right corner.
        
        region_lat_upper_right : float
            Latitude of the upper-right corner.

        output_dir : str
            Directory where images will be saved.

        tile_size : float, optional (default=2)
            Size of each tile in kilometers.

        Notes:
        -----
        - Only images not already present in the output directory are processed.
        - Prints progress for each processed tile.
        """

        bounding_box_array_2d = self.canvas_region_rectangular()
        total_images = len(bounding_box_array_2d) * len(bounding_box_array_2d[0])
        processed = 0
        image_data = []

        # iterate through the 2d array of bounding boxes and download the image for each bounding box
        for row in range(len(bounding_box_array_2d)):
            for col in range(len(bounding_box_array_2d[row])):
                bbox = bounding_box_array_2d[row][col]
                file_name = f'{bbox.lower_left[0]},{bbox.lower_left[1]}'

                filepath = os.path.join(self.data_folder, file_name)

                # check if file already exists before calling sentinel api
                png_filepath = filepath + '.png'
                if not os.path.exists(png_filepath):
                    img = self.get_image_square(bbox.lower_left[0], bbox.lower_left[1])
                    self.save_image(img, png_filepath, factor=3.5, clip_range=(0, 255))

                # create entry for json file to keep track of image locations on map
                json_data = {
                    'image_path' : png_filepath, 
                    "bbox_lower_left" : bbox.lower_left,
                    "bbox_upper_right" : bbox.upper_right
                }

                image_data.append(json_data)
                processed += 1
                print(f'Processed {processed} / {total_images}')

        # write json file
        with open(self.data_folder + "/images.json", "w") as f:
            json.dump(image_data, f, indent=2)

    def save_image(self, image: np.ndarray, filepath: str, factor: float = 1.0, clip_range: tuple[float, float] | None = None, **kwargs: Any):

        if clip_range is not None:
            image = np.clip(image * factor, *clip_range)
        else:
            image = image * factor

        image = np.clip(image, 0, 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV saving
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Save the image using OpenCV
        cv2.imwrite(filepath, image_bgr)


test_path = '/Users/jacksonhayward/Desktop/boulder_river'
data_getter = SentinelTileDownloader(2, lower_left[0], lower_left[1], upper_right[0], upper_right[1], test_path)
data_getter.get_images_from_region()
