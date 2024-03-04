import pyautogui
import time
import keyboard
import cv2 as cv
import numpy as np

market_image = cv.imread('test3.png', cv.COLOR_BGR2GRAY)


def find_market_items(image: np.ndarray) ->list[np.ndarray]:
    ret1, thresh_img = cv.threshold(image, 60, 255, cv.THRESH_TRUNC)
    thresh_img = cv.cvtColor(thresh_img, cv.COLOR_BGR2GRAY)
    
    contours, _ = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    listings = []
    for i in range(1, 11):
        x, y, w, h = cv.boundingRect(contours[i])
        region = image[y: y + h, x: x + w]
        listings.append(region)
    return listings


def find_item_stats():
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)
    screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR)

    edges = cv.Canny(screenshot, threshold1=255, threshold2=255, apertureSize=7 )
    contours, _ = cv.findContours(edges, cv.RETR_TREE,  cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    
    image_area = screenshot.shape[0] * screenshot.shape[1]
    min_area = 0.020 * image_area
    max_area = 1.0 * image_area

    filtered_contours = [cnt for cnt in contours if min_area < cv.contourArea(cnt) < max_area]
    print(len(filtered_contours))

    ## Start here
    rectangles = []
    for cnt in filtered_contours:
        epsilon = 0.1 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            rectangles.append(approx)

    if rectangles:
        x, y, w, h = cv.boundingRect(rectangles[0])
        
        region = screenshot[y: y + h, x: x + w]
        return region
    else:
        return None
    
while True:
    result = find_item_stats()
    
    if result is not None:        
        cv.imshow('result', result)
        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
