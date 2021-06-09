import cv2
import numpy as np
import math
import timeit
from tqdm import tqdm


def coordinates(seeds, p):
    (h, w) = seeds.shape[:2]
    n = 0
    for i in range(h):
        for j in range(w):
            if seeds[i][j][2] == p:
                n += 1
    x = [0]*n
    y = [0]*n
    v = 0
    for i in range(h):
        for j in range(w):
            if seeds[i][j][2] == p:
                x[v] = j
                y[v] = i
                v += 1
    return x, y


def k_means(k, x, y):
    img = cv2.imread('dance.PNG')
    a = len(x)/k
    print('length a:', a)
    c_red = [0] * k
    c_green = [0] * k
    c_blue = [0] * k
    v = 0
    for i in range(0, len(x), int(a)+1):
        #y_cord, x_cord = y[i], x[i]
        r, g, b = img[y[i]][x[i]]
        c_red[v] = r
        c_green[v] = g
        c_blue[v] = b
        v += 1
    clusters = clustering(k, x, y, c_red, c_green, c_blue, img)

    m, itr = 0, 0
    while m == 0:
        print('Iteration: ', itr)
        itr += 1
        w = np.zeros(k)
        updt_red = np.zeros(k)
        updt_green = np.zeros(k)
        updt_blue = np.zeros(k)
        for i in range(k):
            #v = 0
            for j in range(len(x)):
                if clusters[j] == i:
                    r, g, b = img[y[j]][x[j]]
                    updt_red[i] += r
                    updt_green[i] += g
                    updt_blue[i] += b
                    w[i] += 1
        for i in range(k):
            if w[i] != 0:
                updt_red[i] = updt_red[i] / w[i]
                updt_green[i] = updt_green[i] / w[i]
                updt_blue[i] = updt_blue[i] / w[i]
        for i in range(k):
            if abs(c_red[i] - updt_red[i]) > 2 or abs(c_green[i] - updt_green[i]) > 2 or abs(c_blue[i] - updt_blue[i]) > 2:
                clusters = clustering(k, x, y, updt_red, updt_green, updt_blue, img)
                c_red = updt_red
                c_green = updt_green
                c_blue = updt_blue
                break
            elif i == k-1:
                m = 1
    return clusters, updt_red, updt_green, updt_blue


def clustering(k, x, y, c_red, c_green, c_blue, orig):
    clusters = np.array(x)
    for i in range(len(x)):
        a = [0.0] * k
        for j in range(k):
            r, g, b = orig[y[i]][x[i]]
            a[j] = float(math.sqrt(((r - c_red[j])**2) + ((g - c_green[j])**2) + ((b - c_blue[j])**2)))
            if j == (k - 1):
                clusters[i] = (np.argmin(a))
    return clusters


def weight(k, clusters):
    wt = [0.0]*k
    for i in range(k):
        for j in range(len(clusters)):
            if clusters[j] == i:
                wt[i] += 1
            if j == len(clusters) - 1:
                wt[i] = float(wt[i] / len(clusters))
    return wt


def main():
    start = timeit.default_timer()
    seeds = cv2.imread('lady stroke 1.png')
    img = cv2.imread('lady.PNG')

    k = 64
    # Foreground
    print('Foreground')
    front_x, front_y = coordinates(seeds, 255)
    clusters_front, red_front, green_front, blue_front = k_means(k, front_x, front_y)
    # Background
    print('Background')
    back_x, back_y = coordinates(seeds, 6)
    clusters_back, red_back, green_back, blue_back = k_means(k, back_x, back_y)

    print('Weight Foreground')
    weight_front = weight(k, clusters_front)
    print('Weight Background')
    weight_back = weight(k, clusters_back)

    print('Likelihood')
    img_result = img.copy()
    (h, w) = img_result.shape[:2]
    for i in tqdm(range(h)):
        for j in range(w):
            #print(i, j)
            prob_back = 0
            prob_front = 0
            for m in range(k):
                r, g, b = img_result[i][j]
                prob_back += float(weight_back[m] * math.exp(-(math.sqrt(((r - red_back[m])**2) + ((g - green_back[m])**2)
                                                        + ((b - blue_back[m])**2)))))
                prob_front += float(weight_front[m] * math.exp(-(math.sqrt(((r - red_front[m])**2) + ((g - green_front[m])**2)
                                                       + ((b - blue_front[m])**2)))))
                if m == k - 1:
                    if prob_front > prob_back:
                        img_result[i][j][0] = 255
                        img_result[i][j][1] = 255
                        img_result[i][j][2] = 255
                    elif prob_front < prob_back:
                        img_result[i][j][0] = 0
                        img_result[i][j][1] = 0
                        img_result[i][j][2] = 0

    result = cv2.imread('lady.PNG')
    # Segmenting the original image
    print('Segmenting the original image')
    (h, w) = result.shape[:2]
    for i in tqdm(range(h)):
        for j in range(w):
            if img_result[i][j][0] == 0:
                result[i][j] = img_result[i][j]

    cv2.imshow('Segmentation', img_result)
    cv2.waitKey(0)
    cv2.imshow('Segmentation on original', result)
    cv2.waitKey(0)

    end = timeit.default_timer()
    print("Run time", end - start)


if __name__ == "__main__":main()