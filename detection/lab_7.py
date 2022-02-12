import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def homo(img1, img2, sift, flann, cont=None):
    """
    Finds homography and paint over the ghost

    Parameters
    ----------
    img1 image
         query image
    img2 image
         train image
    sift SIFT
    flann FLANN

    Returns
    -------
    img2 image
         result image
    cont np.array
         contours of square
    flag boolean
         True, if finding is successful
    """

    MIN_MATCH_COUNT = 10

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        h, w, d = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        cont = [np.int32(dst)]
        img2 = cv.fillPoly(img2, pts=[np.int32(dst)], color=(255, 255, 255))
        flag = True
    else:
        flag = False
        return img2, cont, flag

    return img2, cont, flag


def main():
    img1 = cv.imread('scary_ghost.png')  # queryImage
    img3 = cv.imread('candy_ghost.png')  # queryImage
    img4 = cv.imread('pampkin_ghost.png')  # queryImage
    imgres = cv.imread('lab7.png')  # trainImage
    img5 = cv.flip(img1, 1)
    default_img = cv.imread('lab7.png')

    FLANN_INDEX_KDTREE = 1

    sift = cv.SIFT_create()

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    for img in [img1, img3, img4, img5]:
        flag = True
        while flag:
            imgres, cont, flag = homo(img, imgres, sift, flann)
            default_img = cv.polylines(default_img, cont, True, [255, 0, 255], 3, cv.LINE_AA)

    default_img = cv.cvtColor(default_img, cv.COLOR_BGR2RGB)
    plt.imshow(default_img)
    plt.show()


if __name__ == '__main__':
    main()
