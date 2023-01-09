import numpy as np
import cv2 as cv

def scaleImage(img, scaleFactor, bShow=False):
    """
    Scaling
    Scaling is just resizing of the image. OpenCV comes with a function cv.resize() for this purpose.
    The size of the image can be specified manually, or you can specify the scaling factor.
    Different interpolation methods are used.
    Preferable interpolation methods are cv.INTER_AREA for shrinking and cv.INTER_CUBIC (slow) & cv.INTER_LINEAR for zooming.
    By default, the interpolation method cv.INTER_LINEAR is used for all resizing purposes.
    You can resize an input image with either of following methods
    :param img: input image
    :param scaleFactor: tuple which is the target dimension after scaling
    :param bShow: whether to show the output result
    :return: scaled image
    """
    scale = cv.resize(img, scaleFactor, interpolation=cv.INTER_CUBIC)
    if bShow:
        cv.imshow('scaled img', scale)
        cv.waitKey(0)
        # cv.destroyAllWindows()
    return scale


def translateImage(img, offsetFactor, bShow=False):
    """
    Translation
    Translation is the shifting of an object's location. If you know the shift in the (x,y) direction and let it be (tx,ty),
     you can create the transformation matrix M as follows:
    M=[1    0   tx
       0    1   ty]
    You can take make it into a Numpy array of type np.float32 and pass it into the cv.warpAffine() function.
    See the below example for a shift of offsetFactor:
    :param img: input image
    :param offsetFactor: tuple to indicate the translatioon offset
    :param bShow: whether to show image or not
    :return: translated image
    """
    rows, cols = img.shape
    M = np.array([[1, 0, offsetFactor[0]],
                  [0, 1, offsetFactor[1]]]).astype("float")
    trans = cv.warpAffine(img, M, (cols, rows))
    if bShow:
        cv.imshow("Translated img", trans)
        cv.waitKey(0)
        # cv.destroyAllWindows()
    return trans

print(translateImage([1,2],[2,2]))
def rotateImage(img, angle, center=(-1, -1), bShow=False):
    """
    Rotation:

    Rotation of an image for an angle θ is achieved by the transformation matrix of the form

    M=[cosθ -sinθ
       sinθ  cosθ]
    But OpenCV provides scaled rotation with adjustable center of rotation so that you can rotate at any location you prefer.
    The modified transformation matrix is given by

    [α  β   (1−α)⋅center.x−β⋅center.y
    -β  α   β⋅center.x+(1−α)⋅center.y]
    where:

    α=scale⋅cosθ,
    β=scale⋅sinθ
    To find this transformation matrix, OpenCV provides a function,
    cv.getRotationMatrix2D. Check out the below example which rotates the image by
    angle in degree with respect to center without any scaling.
    :param img:
    :param angle:
    :param anchorPt:
    :param bShow:
    :return: rotated image
    """
    rows, cols = img.shape
    # cols-1 and rows-1 are the coordinate limits.
    if center[0] == -1:
        center = ((cols - 1) / 2.0,(rows - 1) / 2.0)
    M = cv.getRotationMatrix2D(center, angle, 1)
    print(M)
    rot = cv.warpAffine(img, M, (cols, rows))
    if bShow:
        cv.imshow("rotated img", rot)
        cv.waitKey(0)
        # cv.destroyAllWindows()
    return rot


def affineImage(img, pts1, pts2, bShow=False):
    """
    Affine Transformation
    In affine transformation, all parallel lines in the original image will still be parallel in the output image.
    To find the transformation matrix, we need three points from the input image and their corresponding locations in the output image.
    Then cv.getAffineTransform will create a 2x3 matrix which is to be passed to cv.warpAffine.

    Check the below example, and also look at the points I selected (which are marked in green color):
    :param img: input image
    :param pts1: list of reference points in img. exp: pts1 = np.float32([[50,50],[200,50],[50,200]])
    :param pts2: list of projected points in affine image: pts2 = np.float32([[10,100],[200,50],[100,250]])
    :param bShow:
    :return: affine image
    """
    rows, cols, ch = img.shape
    M = cv.getAffineTransform(pts1, pts2)
    affine = cv.warpAffine(img, M, (cols, rows))
    if bShow:
        cv.imshow("affine Image", affine)
        cv.waitKey(0)
    return affine


def perspectiveImage(img, pts1, pts2, bShow=False):
    """
    Perspective Transformation
    For perspective transformation, you need a 3x3 transformation matrix.
    Straight lines will remain straight even after the transformation.
    To find this transformation matrix, you need 4 points on the input image and corresponding points on the output image.
    Among these 4 points, 3 of them should not be collinear.
    Then the transformation matrix can be found by the function cv.getPerspectiveTransform.
    Then apply cv.warpPerspective with this 3x3 transformation matrix.
    :param img: input image
    :param pts1: 4 reference points in img ( 3 of them should not be collinear ). exp: pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    :param pts2: 4 projection points in result. exp: pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    :param bShow: whether show new image or not
    :return: perpective image
    """
    M = cv.getPerspectiveTransform(pts1, pts2)
    per = cv.warpPerspective(img, M, (300, 300))
    if bShow:
        cv.imshow("perspective image", per)
        cv.waitKey(0)


def scaleImageDemo():
    """
    Demo for scaling image func.
    Read an image using cv.imread() and
    then zoomin the image by 2
    :return: None
    """
    img = cv.imread('messi5.jpeg')
    cv.imshow("before scale", img)
    scaleImage(img, (2, 2), True)
    cv.destroyAllWindows()


def translateImageDemo():
    """
    Demo shift image using translateImage func.
    Read an image using cv.imread() and
    then shift (100,50) pixels the original
    image
    :return: None
    """
    img = cv.imread('messi5.jpeg', 0)
    cv.imshow("before translate", img)
    offsetFactor = (50,100)
    translateImage(img, offsetFactor, True)
    cv.destroyAllWindows()


def rotateImageDemo():
    """
    Demo rotation using rotateImage func.
    Read an image using cv.imread(0 and then
    rotate the image by 60 degree
    :return: None
    """
    img = cv.imread('messi5.jpeg', 0)
    cv.imshow("before rotate", img)
    rotateImage(img,60, (-1, -1), True)
    cv.destroyAllWindows()


def affineImageDemo():
    """
    Demo affine an image by using affineImage func.
    Read an image using cv.imread() and then
    affine Image
    :return: None
    """
    img = cv.imread('sudoku.jpeg')
    cv.imshow("before rotate", img)
    pts1 = np.float32([[153, 163], [264, 160], [153, 272]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    affineImage(img, pts1, pts2, True)
    cv.destroyAllWindows()


def perspectiveImageDemo():
    """
    Demo perspective image by using perspectiveImage func
    read an image using cv.imread() and then perspective them
    :return:
    """
    img = cv.imread('sudoku.jpeg')
    cv.imshow("before perspective", img)
    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    perspectiveImage(img, pts1, pts2, True)
    cv.destroyAllWindows()

def main():
    # translateImageDemo()
    rotateImageDemo()
    # affineImageDemo()
    #perspectiveImageDemo()

#if __name__ == "main":
