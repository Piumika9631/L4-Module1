import os
from typing import TypeVar, NamedTuple, List, Optional, Tuple
import cv2
import imutils
import numpy as np
from scipy.spatial import ConvexHull
from .arg_parser import SwtArgParser
from .id_colors import build_colormap

Image = np.ndarray
GradientImage = np.ndarray
Position = NamedTuple('Position', [('x', int), ('y', int)])
Stroke = NamedTuple('Stroke', [('x', int), ('y', int), ('width', float)])
Ray = List[Position]
Component = List[Position]
ImageOrValue = TypeVar('ImageOrValue', float, Image)
Gradients = NamedTuple('Gradients', [('x', GradientImage), ('y', GradientImage)])


def gamma(x: ImageOrValue, coeff: float = 2.2) -> ImageOrValue:
    """
    Applies a gamma transformation to the input.

    :param x: The value to transform.
    :param coeff: The gamma coefficient to use.
    :return: The transformed value.
    """
    return x ** (1. / coeff)


def gleam(im: Image, gamma_coeff: float = 2.2) -> Image:
    """
    Implements Gleam grayscale conversion from
    Kanan & Cottrell 2012: Color-to-Grayscale: Does the Method Matter in Image Recognition?
    http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0029740

    :param im: The image to convert.
    :param gamma_coeff: The gamma coefficient to use.
    :return: The grayscale converted image.
    """
    im = gamma(im, gamma_coeff)
    im = np.mean(im, axis=2)
    return np.expand_dims(im, axis=2)


def open_grayscale(path: str) -> Image:
    """
    Opens an image and converts it to grayscale.

    :param path: The image to open.
    :return: The grayscale image.
    """
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    resized_image = resize_image(im, height=1200)
    # cv2.imshow('resized_image', resized_image)
    # extract the envelop
    envelop_marked_obj = get_largest_element(resized_image)
    # cv2.imshow('envelop', envelop_marked_obj['envelop'])
    # remove background
    ROI = apply_background_mask(resized_image, envelop_marked_obj)
    # cv2.imshow('ROI', ROI)
    # skew correction
    rotated_image = correct_skewness(ROI)
    largest_obj = get_largest_element(rotated_image)
    cropped_image = background_crop(rotated_image, largest_obj);
    cv2.imshow('cropped_image', cropped_image)

    processed_image = cropped_image.astype(np.float32) / 255.
    return gleam(processed_image)


def correct_skewness(image):
    envelop_obj = get_largest_element(image)
    angle = cv2.minAreaRect(envelop_obj['contour'])[-1]

    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make it positive
    else:
        angle = -angle

    rotated = rotate_bound(image, angle)  # imutils.rotate_bound(image, angle)
    # rotated = imutils.rotate_bound(image, angle)

    # draw the correction angle on the image so we can validate it
    # cv.putText(rotated, "Angle: {:.2f} degrees".format(angle),
    #            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # show the output image
    # print("[INFO] angle: {:.3f}".format(angle))

    # rotated_resized_image = resize_image(rotated, height=700)

    return rotated


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC)


def apply_background_mask(image, largest_element):
    image_copy = image.copy()
    # a mask is the same size as our image, but has only two pixel values,
    # 0 and 255 -- pixels with a value of 0 (background) are ignored in the original image
    # while mask pixels with a value of 255 (foreground) are allowed to be kept

    mask = np.zeros(image.shape[:2], np.uint8)
    c = largest_element['contour']
    cv2.drawContours(mask, [c], -1, 255, -1)
    masked_image = cv2.bitwise_and(image_copy, image_copy, mask=mask)
    return masked_image


def background_crop(image, largest_element):
    image_copy = image.copy()
    envelop_contour = largest_element['contour']
    x, y, w, h = cv2.boundingRect(envelop_contour)
    crop_img = image_copy[y:y + h, x:x + w]
    return crop_img
    # cv2.waitKey(0)


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dimension = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dimension = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dimension = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dimension, interpolation=inter)

    # return the resized image
    return resized


def get_largest_element(original_image):
    image_copy = original_image.copy()
    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(original_image, contours, -1, (0, 255, 0), 1)
    contours_list = []

    for i in contours:
        M = cv2.contourArea(i)
        my_object = {'contour': i, 'area': M}
        # print('AREA: ' + str(M))
        contours_list.append(my_object)

    max = 0
    max_obj = contours_list[0]
    for obj in contours_list:
        if obj['area'] > max:
            max = obj['area']
            max_obj = obj

    # print('MAX AREA: ' + str(max))
    # cv2.drawContours(original_image, max_obj['contour'], -1, (0, 0, 255), 2)
    max_obj['envelop'] = original_image
    return max_obj


def get_edges(im: Image, lo: float = 175, hi: float = 220, window: int = 3) -> Image:
    """
    Detects edges in the image by applying a Canny edge detector.

    :param im: The image.
    :param lo: The lower threshold.
    :param hi: The higher threshold.
    :param window: The window (aperture) size.
    :return: The edges.
    """
    # OpenCV's Canny detector requires 8-bit inputs.
    im = (im * 255.).astype(np.uint8)
    edges = cv2.Canny(im, lo, hi, apertureSize=window)
    # Note that the output is either 255 for edges or 0 for other pixels.
    # Conversion to float wastes space, but makes the return value consistent
    # with the other methods.
    return edges.astype(np.float32) / 255.


def get_gradients(im: Image) -> Gradients:
    """
    Obtains the image gradients by means of a 3x3 Scharr filter.

    :param im: The image to process.
    :return: The image gradients.
    """
    # In 3x3, Scharr is a more correct choice than Sobel. For higher
    # dimensions, Sobel should be used.
    grad_x = cv2.Scharr(im, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(im, cv2.CV_64F, 0, 1)
    return Gradients(x=grad_x, y=grad_y)


def get_gradient_directions(g: Gradients) -> Image:
    """
    Obtains the gradient directions.

    :param g: The gradients.
    :return: An image of the gradient directions.
    """
    return np.arctan2(g.y, g.x)


def apply_swt(im: Image, edges: Image, gradients: Gradients, dark_on_bright: bool = True) -> Image:
    """
    Applies the Stroke Width Transformation to the image.

    :param im: The image
    :param edges: The edges of the image.
    :param gradients: The gradients of the image.
    :param dark_on_bright: Enables dark-on-bright text detection.
    :return: The transformed image.
    """
    # Prepare the output map.
    swt = np.squeeze(np.ones_like(im)) * np.Infinity

    # For each pixel, let's obtain the normal direction of its gradient.
    norms = np.sqrt(gradients.x ** 2 + gradients.y ** 2)
    norms[norms == 0] = 1
    inv_norms = 1. / norms
    directions = Gradients(x=gradients.x * inv_norms, y=gradients.y * inv_norms)

    # We keep track of all the rays found in the image.
    rays = []

    # Find a pixel that lies on an edge.
    height, width = im.shape[0:2]
    for y in range(height):
        for x in range(width):
            # Edges are either 0. or 1.
            if edges[y, x] < .5:
                continue
            ray = swt_process_pixel(Position(x=x, y=y), edges, directions, out=swt, dark_on_bright=dark_on_bright)
            if ray:
                rays.append(ray)

    # Multiple rays may cross the same pixel and each pixel has the smallest
    # stroke width of those.
    # A problem are corners like the edge of an L. Here, two rays will be found,
    # both of which are significantly longer than the actual width of each
    # individual stroke. To mitigate, we will visit each pixel on each ray and
    # take the median stroke length over all pixels on the ray.
    for ray in rays:
        median = np.median([swt[p.y, p.x] for p in ray])
        for p in ray:
            swt[p.y, p.x] = min(median, swt[p.y, p.x])

    swt[swt == np.Infinity] = 0
    return swt


def swt_process_pixel(pos: Position, edges: Image, directions: Gradients, out: Image, dark_on_bright: bool = True) -> \
        Optional[Ray]:
    """
    Obtains the stroke width starting from the specified position.
    :param pos: The starting point
    :param edges: The edges.
    :param directions: The normalized gradients
    :param out: The output image.
    :param dark_on_bright: Enables dark-on-bright text detection.
    """
    # Keep track of the image dimensions for boundary tests.
    height, width = edges.shape[0:2]

    # The direction in which we travel the gradient depends on the type of text
    # we want to find. For dark text on light background, follow the opposite
    # direction (into the dark are); for light text on dark background, follow
    # the gradient as is.
    gradient_direction = -1 if dark_on_bright else 1

    # Starting from the current pixel we will shoot a ray into the direction
    # of the pixel's gradient and keep track of all pixels in that direction
    # that still lie on an edge.
    ray = [pos]

    # Obtain the direction to step into
    dir_x = directions.x[pos.y, pos.x]
    dir_y = directions.y[pos.y, pos.x]

    # Since some pixels have no gradient, normalization of the gradient
    # is a division by zero for them, resulting in NaN. These values
    # should not bother us since we explicitly tested for an edge before.
    assert not (np.isnan(dir_x) or np.isnan(dir_y))

    # Traverse the pixels along the direction.
    prev_pos = Position(x=-1, y=-1)
    steps_taken = 0
    while True:
        # Advance to the next pixel on the line.
        steps_taken += 1
        cur_x = int(np.floor(pos.x + gradient_direction * dir_x * steps_taken))
        cur_y = int(np.floor(pos.y + gradient_direction * dir_y * steps_taken))
        cur_pos = Position(x=cur_x, y=cur_y)
        if cur_pos == prev_pos:
            continue
        prev_pos = Position(x=cur_x, y=cur_y)
        # If we reach the edge of the image without crossing a stroke edge,
        # we discard the result.
        if not ((0 <= cur_x < width) and (0 <= cur_y < height)):
            return None
        # The point is either on the line or the end of it, so we register it.
        ray.append(cur_pos)
        # If that pixel is not an edge, we are still on the line and
        # need to continue scanning.
        if edges[cur_y, cur_x] < .5:  # TODO: Test for image boundaries here
            continue
        # If this edge is pointed in a direction approximately opposite of the
        # one we started in, it is approximately parallel. This means we
        # just found the other side of the stroke.
        # The original paper suggests the gradients need to be opposite +/- PI/6.
        # Since the dot product is the cosine of the enclosed angle and
        # cos(pi/6) = 0.8660254037844387, we can discard all values that exceed
        # this threshold.
        cur_dir_x = directions.x[cur_y, cur_x]
        cur_dir_y = directions.y[cur_y, cur_x]
        dot_product = dir_x * cur_dir_x + dir_y * cur_dir_y
        if dot_product >= -0.866:
            return None
        # Paint each of the pixels on the ray with their determined stroke width
        stroke_width = np.sqrt((cur_x - pos.x) * (cur_x - pos.x) + (cur_y - pos.y) * (cur_y - pos.y))
        for p in ray:
            out[p.y, p.x] = min(stroke_width, out[p.y, p.x])
        return ray

    # noinspection PyUnreachableCode
    assert False, 'This code cannot be reached.'


def connected_components(swt: Image, threshold: float = 3.) -> Tuple[Image, List[Component]]:
    """
    Applies Connected Components labeling to the transformed image using a flood-fill algorithm.

    :param swt: The Stroke Width transformed image.
    :param threshold: The Stroke Width ratio below which two strokes are considered the same.
    :return: The map of labels.
    """
    height, width = swt.shape[0:2]
    labels = np.zeros_like(swt, dtype=np.uint32)
    next_label = 0
    components = []  # List[Component]
    for y in range(height):
        for x in range(width):
            stroke_width = swt[y, x]
            if (stroke_width <= 0) or (labels[y, x] > 0):
                continue
            next_label += 1
            neighbor_labels = [Stroke(x=x, y=y, width=stroke_width)]
            component = []
            while len(neighbor_labels) > 0:
                neighbor = neighbor_labels.pop()
                npos, stroke_width = Position(x=neighbor.x, y=neighbor.y), neighbor.width
                if not ((0 <= npos.x < width) and (0 <= npos.y < height)):
                    continue
                # If the current pixel was already labeled, skip it.
                n_label = labels[npos.y, npos.x]
                if n_label > 0:
                    continue
                # We associate pixels based on their stroke width. If there is no stroke, skip the pixel.
                n_stroke_width = swt[npos.y, npos.x]
                if n_stroke_width <= 0:
                    continue
                # We consider this point only if it is within the acceptable threshold and in the initial test
                # (i.e. when visiting a new stroke), the ratio is 1.
                # If we succeed, we can label this pixel as belonging to the same group. This allows for
                # varying stroke widths due to e.g. perspective distortion or elaborate fonts.
                if (stroke_width / n_stroke_width >= threshold) or (n_stroke_width / stroke_width >= threshold):
                    continue
                labels[npos.y, npos.x] = next_label
                component.append(npos)
                # From here, we're going to expand the new neighbors.
                neighbors = {Stroke(x=npos.x - 1, y=npos.y - 1, width=n_stroke_width),
                             Stroke(x=npos.x, y=npos.y - 1, width=n_stroke_width),
                             Stroke(x=npos.x + 1, y=npos.y - 1, width=n_stroke_width),
                             Stroke(x=npos.x - 1, y=npos.y, width=n_stroke_width),
                             Stroke(x=npos.x + 1, y=npos.y, width=n_stroke_width),
                             Stroke(x=npos.x - 1, y=npos.y + 1, width=n_stroke_width),
                             Stroke(x=npos.x, y=npos.y + 1, width=n_stroke_width),
                             Stroke(x=npos.x + 1, y=npos.y + 1, width=n_stroke_width)}
                neighbor_labels.extend(neighbors)
            if len(component) > 0:
                components.append(component)
    return labels, components


def minimum_area_bounding_box(points: np.ndarray) -> np.ndarray:
    """
    Determines the minimum area bounding box for the specified set of points.

    :param points: The point coordinates.
    :return: The coordinates of the bounding box.
    """
    # The minimum area bounding box is aligned with at least one
    # edge of the convex hull. (TODO: Proof?)
    # This reduces the number of orientations we have to try.
    hull = ConvexHull(points)
    for i in range(len(hull.vertices) - 1):
        # Select two vertex pairs and obtain their orientation to the X axis.
        a = points[hull.vertices[i]]
        b = points[hull.vertices[i + 1]]
        # TODO: Find orientation. Note that sine = abs(cross product) and cos = dot product of two vectors.
        print(a, b)
    return points


def discard_non_text(swt: Image, labels: Image, components: List[Component]) -> Tuple[Image, List[Component]]:
    """
    Discards components that are likely not text.
    
    :param swt: The stroke-width transformed image.
    :param labels: The labeled components.
    :param components: A list of each component with all its pixels.
    :return: The filtered labels and components.
    """
    invalid_components = []  # type: List[Component]
    for component in components:
        # If the variance of the stroke widths in the component is more than
        # half the average of the stroke widths of that component, it is considered invalid.
        average_stroke = np.mean([swt[p.y, p.x] for p in component])
        variance = np.var([swt[p.y, p.x] for p in component])
        if variance > .5 * average_stroke:
            invalid_components.append(component)
            continue
        # Natural scenes may create very long, yet narrow components. We prune
        # these based on their aspect ratio.
        points = np.array([[p.x, p.y] for p in component], dtype=np.uint32)
        minimum_area_bounding_box(points)
        print(variance)
    return labels, components


def remove_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_r = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    edges = cv2.Canny(thresh_r, 50, 200)
    lines = cv2.HoughLinesP(edges, rho=1, theta=1 * np.pi / 180,
                            threshold=10, minLineLength=20, maxLineGap=3)
    # Draw lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv2.imshow('rexsult', image)


def main():
    # swt = np.array(
    # [
    #    [14, 1, 4, 4, 0],
    #    [14, 1, 4, 1, 1],
    #    [14, 1, 4, 1, 0],
    #    [14, 1, 1, 1, 0],
    #    [ 4, 0, 4, 0, 0]
    # ], dtype=np.float32)

    # labels = connected_components(swt)
    # l = (labels / labels.max() * 255.).astype(np.uint8)
    # swt = (swt / swt.max() * 255.).astype(np.uint8)
    # cv2.imwrite('swt.png', swt)
    # cv2.imwrite('comps.png', l)
    # return

    imagePath = 'images/letter2.jpg'

    parser = SwtArgParser()
    args = parser.parse_args()
    args.image = imagePath
    if not os.path.exists(args.image):
        parser.error('Image file does not exist: {}'.format(args.image))

    # Open the image and obtain a grayscale representation.
    im = open_grayscale(args.image)  # TODO: Magic numbers hidden in arguments

    # Find the edges in the image and the gradients.
    edges = get_edges(im)  # TODO: Magic numbers hidden in arguments
    gradients = get_gradients(im)  # TODO: Magic numbers hidden in arguments

    # TODO: Gradient directions are only required for checking if two edges are in opposing directions. We can use the gradients directly.
    # Obtain the gradient directions. Due to symmetry, we treat opposing
    # directions as the same (e.g. 180° as 0°, 135° as 45°, etc.).
    # theta = get_gradient_directions(gradients)
    # theta = np.abs(theta)

    # Apply the Stroke Width Transformation.
    swt = apply_swt(im, edges, gradients, not args.bright_on_dark)

    # Apply Connected Components labelling
    labels, components = connected_components(swt)  # TODO: Magic numbers hidden in arguments

    # Discard components that are likely not text
    # TODO: labels, components = discard_non_text(swt, labels, components)

    labels = labels.astype(np.float32) / labels.max()
    l = (labels * 255.).astype(np.uint8)

    l = cv2.cvtColor(l, cv2.COLOR_GRAY2RGB)
    l = cv2.LUT(l, build_colormap())
    cv2.imwrite('comps.png', l)

    swt = (255 * swt / swt.max()).astype(np.uint8)
    cv2.imwrite('swt.png', swt)

    # cv2.imshow('Image', im)
    # cv2.imshow('Edges', edges)
    # cv2.imshow('X', gradients.x)
    # cv2.imshow('Y', gradients.y)
    # cv2.imshow('Theta', theta)
    # cv2.imshow('Stroke Width Transformed', swt)
    cv2.imshow('Connected Components', l)

    # START HERE
    result_r = l.copy()
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(result_r, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    # Convert the image to gray scale
    gray_r = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)

    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray_r, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Appplying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    cv2.imshow("gray_r", gray_r)
    cv2.imshow("dilation", dilation)

    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    result_r2 = l.copy()

    white_board = np.full(result_r2.shape[:2], 255, np.uint8)

    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    # Extracted text is then written into the text file
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if (area > 100 and area < 1000):
            # Drawing a rectangle on copied image
            rect = cv2.rectangle(white_board, (x, y), (x + w, y + h), (0, 255, 0), -1)
        # rect = cv2.rectangle(result_r2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("sdsdf", white_board)
    #hsv = cv2.cvtColor(white_board, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([white_board], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.imshow("hist", hist)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
