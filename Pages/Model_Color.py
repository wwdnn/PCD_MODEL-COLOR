# make code conversion rgb to cmy and cmy to rgb

import cv2
import numpy as np
import matplotlib.pyplot as plt

# function to convert rgb to xyz
def RGB2XYZ(image):
    # get image shape
    height, width, channel = image.shape
    # create new image
    new_image = np.zeros((height, width, channel), dtype=np.uint8)
    # convert rgb to xyz
    for i in range(height):
        for j in range(width):
            # get rgb value
            r = image[i, j, 0]
            g = image[i, j, 1]
            b = image[i, j, 2]
            # convert rgb to xyz
            # 0,49/0,17697 * r + 0,31/0,17697 * g + 0,2/0,17697 * b
            x = 0.49 * r + 0.31 * g + 0.2 * b
            # 0,17697/0,17697 * r + 0,8124/0,17697 * g + 0,01063/0,17697 * b
            y = 0.17697 * r + 0.8124 * g + 0.01063 * b
            # 0,01/0,17697 * r + 0,01/0,17697 * g + 0,99/0,17697 * b
            z = 0.01 * r + 0.01 * g + 0.99 * b

            # div  0,17697
            x = x / 0.17697
            y = y / 0.17697
            z = z / 0.17697

            # assign xyz value to new image
            new_image[i, j, 0] = x
            new_image[i, j, 1] = y
            new_image[i, j, 2] = z
    return new_image

# function to convert rgb to xyz
def RGB2XYZ(r,g,b):
    # convert rgb to xyz
    # 0,49/0,17697 * r + 0,31/0,17697 * g + 0,2/0,17697 * b
    x = 0.49 * r + 0.31 * g + 0.2 * b
    # 0,17697/0,17697 * r + 0,8124/0,17697 * g + 0,01063/0,17697 * b
    y = 0.17697 * r + 0.8124 * g + 0.01063 * b
    # 0,01/0,17697 * r + 0,01/0,17697 * g + 0,99/0,17697 * b
    z = 0.01 * r + 0.01 * g + 0.99 * b

    # div  0,17697
    x = x / 0.17697
    y = y / 0.17697
    z = z / 0.17697

    return x, y, z

# function to convert rgb to CIE-Lab
def RGB2CIELab(image):
    # get image shape
    height, width, channel = image.shape
    # create new image
    new_image = np.zeros((height, width, channel), dtype=np.uint8)

    # xn = 95,047  yn = 100  zn = 108,883
    xn = 95.047
    yn = 100
    zn = 108.883

    # convert rgb to xyz
    for i in range(height):
        for j in range(width):
            # get rgb value
            r = image[i, j, 0]
            g = image[i, j, 1]
            b = image[i, j, 2]
            # convert rgb to xyz
            # 0,49/0,17697 * r + 0,31/0,17697 * g + 0,2/0,17697 * b
            x = 0.49 * r + 0.31 * g + 0.2 * b
            # 0,17697/0,17697 * r + 0,8124/0,17697 * g + 0,01063/0,17697 * b
            y = 0.17697 * r + 0.8124 * g + 0.01063 * b
            # 0,01/0,17697 * r + 0,01/0,17697 * g + 0,99/0,17697 * b
            z = 0.01 * r + 0.01 * g + 0.99 * b

            # div  0,17697
            x = x / 0.17697
            y = y / 0.17697
            z = z / 0.17697

            # fx = x / xn
            # fy = y / yn
            # fz = z / zn
            fx = x / xn
            fy = y / yn
            fz = z / zn

            # if fx > 0,008856
            if fx > 0.008856:
                fx = fx ** (1 / 3)
            else:
                fx = 7.787 * fx + 16 / 116
            # if fy > 0,008856
            if fy > 0.008856:
                fy = fy ** (1 / 3)
            else:
                fy = 7.787 * fy + 16 / 116
            # if fz > 0,008856
            if fz > 0.008856:
                fz = fz ** (1 / 3)
            else:
                fz = 7.787 * fz + 16 / 116

            # L = 116 * fy - 16
            # a = 500 * (fx - fy)
            # b = 200 * (fy - fz)
            L = 116 * fy - 16
            a = 500 * (fx - fy)
            b = 200 * (fy - fz)

            # assign xyz value to new image
            new_image[i, j, 0] = L
            new_image[i, j, 1] = a
            new_image[i, j, 2] = b

    return new_image

# function to convert rgb to CIE-Luv
def RGB2CIELuv(image):
    # get image shape
    height, width, channel = image.shape
    # create new image
    new_image = np.zeros((height, width, channel), dtype=np.uint8)

    # xn = 95,047  yn = 100  zn = 108,883
    xn = 95.047
    yn = 100
    zn = 108.883

    # convert rgb to xyz
    for i in range(height):
        for j in range(width):
            # get rgb value
            r = image[i, j, 0]
            g = image[i, j, 1]
            b = image[i, j, 2]
            # convert rgb to xyz
            # 0,49/0,17697 * r + 0,31/0,17697 * g + 0,2/0,17697 * b
            x = 0.49 * r + 0.31 * g + 0.2 * b
            # 0,17697/0,17697 * r + 0,8124/0,17697 * g + 0,01063/0,17697 * b
            y = 0.17697 * r + 0.8124 * g + 0.01063 * b
            # 0,01/0,17697 * r + 0,01/0,17697 * g + 0,99/0,17697 * b
            z = 0.01 * r + 0.01 * g + 0.99 * b

            # div  0,17697
            x = x / 0.17697
            y = y / 0.17697
            z = z / 0.17697

            #  u' = 4 * x / (x + 15 * y + 3 * z)
            #  v' = 9 * y / (x + 15 * y + 3 * z)
            up = 4 * x / (x + 15 * y + 3 * z)
            vp = 9 * y / (x + 15 * y + 3 * z)
            upn = 4 * xn / (xn + 15 * yn + 3 * zn)
            vpn = 9 * yn / (xn + 15 * yn + 3 * zn)

            # if y/yn > 0,008856
            if y / yn > 0.008856:
                L = 116 * (y / yn) ** (1 / 3) - 16
            else:
                L = 903.3 * (y / yn)

            # u = 13 * L * (u' - upn)
            # v = 13 * L * (v' - vpn)
            u = 13 * L * (up - upn)
            v = 13 * L * (vp - vpn)

            # assign xyz value to new image
            new_image[i, j, 0] = L
            new_image[i, j, 1] = u
            new_image[i, j, 2] = v

    return new_image

# function to convert rgb to ycbcr
def RGB2YCbCr(image):
    # get image shape
    height, width, channel = image.shape
    # create new image
    new_image = np.zeros((height, width, channel), dtype=np.uint8)
    # convert rgb to ycbcr
    for i in range(height):
        for j in range(width):
            # get rgb value
            r = image[i, j, 0]
            g = image[i, j, 1]
            b = image[i, j, 2]
            # convert rgb to ycbcr
            y = 0.299 * r + 0.587 * g + 0.114 * b
            cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
            cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
            # assign ycbcr value to new image
            new_image[i, j, 0] = y
            new_image[i, j, 1] = cb
            new_image[i, j, 2] = cr
    return new_image

# function to convert rgb to ntsc 
def RGB2NTSC(image):
    # get image shape
    height, width, channel = image.shape
    # create new image
    new_image = np.zeros((height, width, channel), dtype=np.uint8)
    # convert rgb to ntsc
    for i in range(height):
        for j in range(width):
            # get rgb value
            r = image[i, j, 0]
            g = image[i, j, 1]
            b = image[i, j, 2]
            # convert rgb to ntsc
            y = 0.299 * r + 0.587 * g + 0.114 * b
            i_ntsc = 0.596 * r - 0.274 * g - 0.322 * b
            q = 0.211 * r - 0.523 * g + 0.312 * b
            # assign ntsc value to new image
            new_image[i, j, 0] = y
            new_image[i, j, 1] = i_ntsc
            new_image[i, j, 2] = q
    return new_image



# function to convert rgb to yuv
def RGB2YUV(image):
    # get image shape
    height, width, channel = image.shape
    # create new image
    new_image = np.zeros((height, width, channel), dtype=np.uint8)
    # convert rgb to yuv
    for i in range(height):
        for j in range(width):
            # get rgb value
            r = image[i, j, 0]
            g = image[i, j, 1]
            b = image[i, j, 2]
            # convert rgb to yuv
            y = 0.299 * r + 0.587 * g + 0.114 * b
            u = -0.147 * r - 0.289 * g + 0.436 * b
            v = 0.615 * r - 0.515 * g - 0.1 * b
            # assign yuv value to new image
            new_image[i, j, 0] = y
            new_image[i, j, 1] = u
            new_image[i, j, 2] = v
    return new_image

# function to convert rgb to hsv
def RGB2HSV(image):
    # get image shape
    height, width, channel = image.shape
    # create new image
    new_image = np.zeros((height, width, channel), dtype=np.uint8)
    # convert rgb to hsv
    for i in range(height):
        for j in range(width):
            # get rgb value
            r = image[i, j, 0]
            g = image[i, j, 1]
            b = image[i, j, 2]
    
            cmax = max(r, g, b)
            cmin = min(r, g, b)

            v = cmax
            vm = v - cmin

            if v == 0:
                s = 0
            else:
                s = vm / v
            
            if s == 0:
                h = 0
            elif cmax == r:
                h = 60 * (((g - b) / vm) % 6)
            elif cmax == g:
                h = 60 * (((b - r) / vm) + 2)
            elif cmax == b:
                h = 60 * (((r - g) / vm) + 4)
           
            # assign hsv value to new image
            new_image[i, j, 0] = h
            new_image[i, j, 1] = s
            new_image[i, j, 2] = v
    return new_image

# function to convert rgb to cmy
def RGB2CMY(image):
    # get image shape
    height, width, channel = image.shape
    # create new image
    new_image = np.zeros((height, width, channel), dtype=np.uint8)
    # convert rgb to cmy
    for i in range(height):
        for j in range(width):
            new_image[i, j, 0] = 255 - image[i, j, 0]
            new_image[i, j, 1] = 255 - image[i, j, 1]
            new_image[i, j, 2] = 255 - image[i, j, 2]
    return new_image

# function to convert rgb to HSI 
def RGB2HSI(image):
    # get image shape
    height, width, channel = image.shape
    # create new image
    new_image = np.zeros((height, width, channel), dtype=np.uint8)
    # convert rgb to hsi
    for i in range(height):
        for j in range(width):
            # get rgb value
            r = image[i, j, 0]
            g = image[i, j, 1]
            b = image[i, j, 2]
            # convert rgb to hsi
            a = np.arccos((0.5 * ((r - g) + (r - b))) / np.sqrt((r - g)**2 + (r - b) * (g - b)))

            if g >= b:
                h = a
            else:
                h = 2 * np.pi - a
            
            s = 1 - (3 * (min(r, g, b) / (r + g + b)))

            intensity = (r + g + b) / 3

            # assign hsi value to new image
            new_image[i, j, 0] = h
            new_image[i, j, 1] = s
            new_image[i, j, 2] = intensity

    return new_image

# streamlit
import streamlit as st
# opencv
import cv2
# numpy
import numpy as np
# PIL
from PIL import Image

# read image from upload streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# check if image is uploaded
if uploaded_file is not None:
    # read image
    image = Image.open(uploaded_file)
    # convert image to numpy array
    image = np.array(image)
    # convert image to rgb
    # show image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Choose color space to convert")
    # select color space
    color_space = st.selectbox('Color Space', ('RGB', 'CMY', 'HSI', 'HSV', 'YUV', 'NTSC', 'YCbCr', 'XYZ', 'CIELAB', 'CIELUV'))
    # check if color space is selected
    if color_space is not None:
        # convert image to selected color space
        if color_space == 'RGB':
            result = image
        elif color_space == 'CMY':
            result = RGB2CMY(image)
        elif color_space == 'HSI':
            result = RGB2HSI(image)
        elif color_space == 'HSV':
            result = RGB2HSV(image)
        elif color_space == 'YUV':
            result = RGB2YUV(image)
        elif color_space == 'NTSC':
            result = RGB2NTSC(image)
        elif color_space == 'YCbCr':
            result = RGB2YCbCr(image)
        elif color_space == 'XYZ':
            x, y, z = RGB2XYZ(128,255,100)
            st.write(x, y, z)
        elif color_space == 'CIELAB':
            result = RGB2CIELab(image)
        elif color_space == 'CIELUV':
            result = RGB2CIELuv(image)

        # show result
        st.image(result, caption='Result Image.', use_column_width=True)