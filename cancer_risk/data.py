import argparse
import os
import time

import cv2
import numpy as np
import pydicom
from mpi4py import MPI
from skimage import exposure
from utils.data_utils import (
    Convert,
    new_cropping_single_dist,
    pad_or_crop_single_maxloc,
)

FULL_HEIGHT = 4096
FULL_WIDTH = 3328
PNG_FOLDER = "log/pnglog_{}"


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("core", type=int, help="core for preprocessing")
    args = parser.parse_args()

    # start MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    recv_data = None
    if rank == 0:
        send_data = range(args.core)
        print("process {} scatter data {} to other processes".format(rank, send_data))
    else:
        send_data = None
    i = comm.scatter(send_data, root=0)
    fix = "{}".format(i).zfill(2)

    pngpath = PNG_FOLDER.format(fix)
    pngdir = open(pngpath, "r").read().split("\n")

    pngdir.pop(-1)
    total = len(pngdir)

    # start preprocessing
    count = 0
    for j in pngdir:
        start = time.time()

        local_path = j.split(";")[0]
        new_path = j.split(";")[1]
        dicom_imagelaterality = j.split(";")[2]
        if "[" in j.split(";")[3]:
            dicom_windowcenter = Convert(j.split(";")[3])
        else:
            dicom_windowcenter = [float(j.split(";")[3])]
        if "[" in j.split(";")[4]:
            dicom_windowwidth = Convert(j.split(";")[4])
        else:
            dicom_windowwidth = [float(j.split(";")[4])]

        # read the dicom file
        img_dcm = pydicom.dcmread(local_path)
        try:
            img_array = img_dcm.pixel_array
        except Exception as e:
            print("dicom_problem")
            print(e)
            continue

        # flip the image if it is right laterality
        if dicom_imagelaterality == "R":
            img_array = cv2.flip(img_array, 1)

        # rescale the image based on window center and window width
        img_array = exposure.rescale_intensity(
            img_array,
            in_range=(
                dicom_windowcenter[0] - dicom_windowwidth[0] / 2,
                dicom_windowcenter[0] + dicom_windowwidth[0] / 2,
            ),
        )

        # invert the image if it is MONOCHROME1
        if img_dcm.PhotometricInterpretation == "MONOCHROME1":
            img_array = cv2.bitwise_not(img_array)

        # crop the image based on the distance transform
        maxLoc = new_cropping_single_dist(img_array)
        new_img = pad_or_crop_single_maxloc(
            img_array, maxLoc, full_height=FULL_HEIGHT, full_width=FULL_WIDTH
        )

        # save the image
        if not os.path.exists(os.path.dirname(new_path)):
            os.makedirs(os.path.dirname(new_path))
        if not os.path.exists(new_path):
            cv2.imwrite(new_path, new_img.astype(np.uint16))

        count += 1

        print(str(count), " images are done")
        print("time for processing: ", str(time.time() - start))
