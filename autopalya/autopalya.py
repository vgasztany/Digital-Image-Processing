import cv2
import numpy as np
import math


# A csúszka pozíciója alapján meghívja a kívánt zajt
def call_add_noise(pos):
    global image
    image = cv2.imread('high-way1.jpg', cv2.IMREAD_COLOR)
    image = cv2.resize(image, (1200, 900))
    if pos == 1:
        cv2.namedWindow('Salt and pepper noise')
        image = add_salt_and_pepper_noise(image, 0.05, 0.05)
        cv2.imshow('Salt and pepper noise', image)
        cv2.waitKey(0)
    elif pos == 2:
        cv2.namedWindow('Noisy')
        cv2.createTrackbar('sigma', 'Noisy', 0, 100, on_sigma_change)
        add_additive_noise(0)
        cv2.waitKey(0)


def add_point_noise(img_in, percentage, value):
    noise_res = np.copy(img_in)
    n = int(img_in.shape[0] * img_in.shape[1] * percentage)
    print(n)

    for k in range(1, n):
        i = np.random.randint(0, img_in.shape[1])
        j = np.random.randint(0, img_in.shape[0])
        if img_in.ndim == 2:
            noise_res[j, i] = value
        if img_in.ndim == 3:
            noise_res[j, i] = [value, value, value]
    return noise_res


def add_salt_and_pepper_noise(img_in, percentage_1, percentage_2):
    n = add_point_noise(img_in, percentage_1, 255)  # Só
    n2 = add_point_noise(n, percentage_2, 0)  # Bors
    return n2


def add_additive_noise(sigma_in):
    global image
    b, g, r = cv2.split(image)
    noise = np.zeros(image.shape[:2], np.int16)
    cv2.randn(noise, 0.0, sigma_in)
    b = cv2.add(b, noise, dtype=cv2.CV_8UC1)
    cv2.randn(noise, 0.0, sigma_in)
    g = cv2.add(g, noise, dtype=cv2.CV_8UC1)
    cv2.randn(noise, 0.0, sigma_in)
    r = cv2.add(r, noise, dtype=cv2.CV_8UC1)
    image = cv2.merge([b, g, r])
    cv2.imshow('Noisy', image)


def on_sigma_change(pos):
    add_additive_noise(pos)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def drow_the_lines(img, lines):
    img = np.copy(img)
    l_upper_left = (0, img.shape[0])
    upper_mid = (0, img.shape[0])
    r_upper_right = (0, img.shape[0])
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
            slope = line_categories(x1, y1, x2, y2)
            if slope == 1:
                if x2 >= l_upper_left[0] and y2 <= l_upper_left[1]:
                    l_upper_left = (x2, y2)
            if slope == 0:
                if y1 <= upper_mid[1]:
                    upper_mid = (x1, y1)
                if y2 <= upper_mid[1]:
                    upper_mid = (x2, y2)
            if slope == -1:
                if x1 >= r_upper_right[0] and y1 <= r_upper_right[1]:
                    r_upper_right = (x1, y1)
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    r_roi, g_roi = roi(l_upper_left, upper_mid, r_upper_right)
    return img, r_roi, g_roi


def line_categories(x1, y1, x2, y2):
    global left_line_slopes
    global mid_line_slopes
    global right_line_slopes
    y1 = (-1) * y1
    y2 = (-1) * y2
    if x2-x1 != 0:
        m = ((y2 - y1) / (x2 - x1))
        alpha = math.atan(m)
        alpha = np.rad2deg(alpha)
        if alpha < 0:
            alpha = alpha + 180
        # Az út bal széle
        if 0 <= alpha <= 80:
            left_line_slopes.append(alpha)
            return 1
        # Az út közepe
        if 80 < alpha <= 110:
            mid_line_slopes.append(alpha)
            return 0
        # Az út jobb széle
        else:
            right_line_slopes.append(alpha)
            return -1
    # Az út közepe
    else:
        mid_line_slopes.append(90)
        return 0


# Átlagos hajlásszög kiszámítása, majd egyenesek metszetei a tengelyekkel
def roi(l_upper_left, upper_mid, r_upper_right):
    global left_line_slopes
    global mid_line_slopes
    global right_line_slopes
    try:
        left_line_average = sum(left_line_slopes) / len(left_line_slopes)
        mid_line_average = sum(mid_line_slopes) / len(mid_line_slopes)
        right_line_average = sum(right_line_slopes) / len(right_line_slopes)
        l_lower_left_y = math.tan(math.radians(left_line_average)) * (0 - l_upper_left[0]) + (-1) * l_upper_left[1]
        l_lower_left_y = (-1) * l_lower_left_y
        lower_mid_x = ((-1) * image.shape[0] - (-1) * upper_mid[1]) / math.tan(math.radians(mid_line_average)) + upper_mid[0]
        r_lower_right_y = math.tan(math.radians(right_line_average)) * (image.shape[1] - r_upper_right[0])\
                          + (-1) * r_upper_right[1]
        r_lower_right_y = r_lower_right_y * (-1)
        red_roi = [
            (0, image.shape[0]),
            (0, l_lower_left_y),
            l_upper_left,
            upper_mid,
            (lower_mid_x, image.shape[0])
        ]
        green_roi = [
            (lower_mid_x, image.shape[0]),
            upper_mid,
            r_upper_right,
            (image.shape[1], r_lower_right_y),
            (image.shape[1], image.shape[0])
        ]
        return red_roi, green_roi
    except ZeroDivisionError:
        raise Exception("A program nem találja a sávokat, állítson a Hough-transzformáció paraméterein!")


def coloring(image):
    # Region of Interest meghatározása
    height = image.shape[0]
    width = image.shape[1]

    region_of_interest_verticles = [
        (0, height),
        (0, 3 * (height / 4)),
        (6 * (width / 13), height / 2),
        (4 * (width / 7), height / 2),
        (width, 3 * (height / 4)),
        (width, height)
    ]

    # Kép szürkeárnyalatos verzója
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray_image)
    cv2.waitKey(0)

    # Éldetektálás
    canny_image = cv2.Canny(gray_image, 200, 400)
    cv2.imshow('gray', canny_image)
    cv2.waitKey(0)

    # Élek maszkolása ROI segítségével
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_verticles], np.int32))
    cv2.imshow('gray', cropped_image)
    cv2.waitKey(0)

    # Hough-transzformáció használata a vonalak detektálására, színezéshez szükséges roi-k előállítása:
    lines = cv2.HoughLinesP(cropped_image,
                            rho=4,
                            theta=np.pi / 60,
                            threshold=80,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=25)
    image_with_lines, r_roi, g_roi = drow_the_lines(image, lines)

    # Színezéshez szükséges maszkok előállítása:
    result = image.copy()
    r_poly = np.array([[r_roi]], dtype=np.int32)
    g_poly = np.array([[g_roi]], dtype=np.int32)
    cv2.fillPoly(result, r_poly, (0, 0, 255))
    cv2.imshow('gray', result)
    cv2.waitKey(0)
    cv2.fillPoly(result, g_poly, (0, 255, 0))
    return result


# Kép betöltése
image = cv2.imread('high-way1.jpg', cv2.IMREAD_COLOR)
image = cv2.resize(image, (1200, 900))
cv2.imshow('image', image)
print('Ha szeretné só-bors zajjal megjeleníteni a képet, állítsa a csúszkát az 1-es pozícióba!'
      'Ha szeretné additív zajjal megjeleníteni a képet, állítsa a csúszkát a 2-es pozícióba!')

cv2.createTrackbar('Add noise', 'image', 0, 2, call_add_noise)
cv2.waitKey(0)

# Sáv elválasztók meredeksége:
left_line_slopes = []
mid_line_slopes = []
right_line_slopes = []

# Sávok színezése és megjelenítése
result = coloring(image)

# Kép megjelenítése
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
