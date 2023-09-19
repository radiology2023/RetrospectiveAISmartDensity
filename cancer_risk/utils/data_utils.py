import datetime
import matplotlib.pyplot as plt
import numpy as np
import cv2


def change_date(date_input):
    month2num = {
        "jan": "January",
        "feb": "February",
        "mar": "March",
        "apr": "April",
        "may": "May",
        "jun": "June",
        "jul": "July",
        "aug": "August",
        "sep": "September",
        "oct": "October",
        "nov": "November",
        "dec": "December",
    }
    date_string = date_input[:2] + month2num[date_input[2:-4]] + date_input[-4:]
    # print(date_string)
    date_object = datetime.strptime(date_string, "%d%B%Y")
    return date_object.timestamp()


# 20121107 to 07nov2012
def change_date_fromnum(date_input):
    month2num = {
        "01": "jan",
        "02": "feb",
        "03": "mar",
        "04": "apr",
        "05": "may",
        "06": "jun",
        "07": "jul",
        "08": "aug",
        "09": "sep",
        "10": "oct",
        "11": "nov",
        "12": "dec",
    }
    date_string = date_input[-2:] + month2num[date_input[4:-2]] + date_input[:4]
    return date_string


def date_diff(late_date, early_date):
    return round((change_date(late_date) - change_date(early_date)) / (3600 * 24))


def latest_date(date_list):
    change_date_list = []
    for date in date_list:
        change_date_list.append(change_date(date))
        ind = change_date_list.index(max(change_date_list))
    return date_list[ind]


def vis_array(img):
    plt.figure()
    plt.imshow(img, "gray")
    plt.pause(0.05)


def new_cropping_single_dist(img):
    org_img = img
    opening_it = 5
    kernel_size = (25, 25)
    kernel = np.ones(kernel_size, np.uint8)
    img = img.astype(np.uint8)
    _, img = cv2.threshold(img, img.min(), img.max(), cv2.THRESH_BINARY)

    img = cv2.GaussianBlur(img, kernel_size, 0)
    img = cv2.dilate(img, kernel, 5)

    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=opening_it)
    opening = cv2.copyMakeBorder(
        opening,
        top=1,
        bottom=1,
        left=1,
        right=1,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    opening = opening.astype(np.uint8)

    dist = cv2.distanceTransform(opening, cv2.cv.CV_DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, kernel_size, 0)
    maxLoc = cv2.minMaxLoc(dist)[-1]

    return maxLoc


def pad_or_crop_single_maxloc(img, maxloc, full_height, full_width):
    # add padding to make maxloc be the center of the image
    w = img.shape[1]
    h = img.shape[0]

    if maxloc[0] >= w - maxloc[0]:
        img_new = np.full((h, 2 * maxloc[0]), 0.0)
        img_new[:, :w] = img
    elif maxloc[0] < w - maxloc[0]:
        img_new = np.full((h, 2 * (w - maxloc[0])), 0.0)
        img_new[:, 2 * (w - maxloc[0]) - w :] = img

    img = img_new
    w = img_new.shape[1]
    h = img_new.shape[0]

    if maxloc[1] >= h - maxloc[1]:
        img_new = np.full((2 * maxloc[1], w), 0.0)
        img_new[:h, :] = img
    elif maxloc[1] < h - maxloc[1]:
        img_new = np.full((2 * (h - maxloc[1]), w), 0.0)
        img_new[2 * (h - maxloc[1]) - h :, :] = img

    img = img_new
    w = img.shape[1]
    h = img.shape[0]

    if h > full_height:
        img = img[
            int((h - full_height) / 2) : int((h - full_height) / 2) + full_height, :
        ]
    if w > full_width:
        img = img[:, int((w - full_width) / 2) : int((w - full_width) / 2) + full_width]
    img_new = np.full((full_height, full_width), 0.0)
    img_new[
        int((img_new.shape[0] - img.shape[0]) / 2) : int(
            (img_new.shape[0] - img.shape[0]) / 2
        )
        + img.shape[0],
        int((img_new.shape[1] - img.shape[1]) / 2) : int(
            (img_new.shape[1] - img.shape[1]) / 2
        )
        + img.shape[1],
    ] = img

    return img_new


def Convert(string):
    li = [float(i) for i in list(string.split("[")[1].split("]")[0].split(", "))]
    return li
