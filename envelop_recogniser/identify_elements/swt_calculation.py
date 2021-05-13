from typing import TypeVar, NamedTuple, List, Optional, Tuple
import cv2 as cv
import numpy as np
from scipy.spatial import ConvexHull
from envelop_recogniser.id_colors import build_colormap

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


def open_grayscale(im: Image) -> Image:
    # Opens an image and converts it to grayscale.
    processed_image = im.astype(np.float32) / 255.
    return gleam(processed_image)


def get_edges(im: Image, lo: float = 175, hi: float = 220, window: int = 3) -> Image:
    # Detects edges in the image by applying a Canny edge detector.
    # OpenCV's Canny detector requires 8-bit inputs.
    im = (im * 255.).astype(np.uint8)
    edges = cv.Canny(im, lo, hi, apertureSize=window)
    # Note that the output is either 255 for edges or 0 for other pixels.
    # Conversion to float wastes space, but makes the return value consistent
    # with the other methods.
    return edges.astype(np.float32) / 255.


def get_gradients(im: Image) -> Gradients:
    # Obtains the image gradients by means of a 3x3 Scharr filter.
    # In 3x3, Scharr is a more correct choice than Sobel. For higher
    # dimensions, Sobel should be used.
    grad_x = cv.Scharr(im, cv.CV_64F, 1, 0)
    grad_y = cv.Scharr(im, cv.CV_64F, 0, 1)
    return Gradients(x=grad_x, y=grad_y)


def get_gradient_directions(g: Gradients) -> Image:
    """
    Obtains the gradient directions.

    :param g: The gradients.
    :return: An image of the gradient directions.
    """
    return np.arctan2(g.y, g.x)


def apply_swt(im: Image, edges: Image, gradients: Gradients, dark_on_bright: bool = True) -> Image:
    # Apply the Stroke Width Transformation to the image
    # dark_on_bright: Enables dark-on-bright text detection

    # Prepare the output map.
    # return image shape array in ones
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
    # Applies Connected Components labeling to the transformed image using a flood-fill algorithm.
    # threshold: The Stroke Width ratio below which two strokes are considered the same
    # return: The map of labels
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
    # Discards components that are likely not text.
    # param components: A list of each component with all its pixels
    # return: The filtered labels and components
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
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh_r = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    edges = cv.Canny(thresh_r, 50, 200)
    lines = cv.HoughLinesP(edges, rho=1, theta=1 * np.pi / 180,
                           threshold=10, minLineLength=20, maxLineGap=3)
    # Draw lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv.imshow('result', image)


def swt_main(image):
    gray = open_grayscale(image)
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Find the edges in the image and the gradients.
    edges = get_edges(gray)
    # Find image gradients
    gradients = get_gradients(gray)

    # Apply the Stroke Width Transformation.
    swt = apply_swt(gray, edges, gradients)

    # Apply Connected Components labelling
    labels, components = connected_components(swt)

    # Discard components that are likely not text
    # TODO: labels, components = discard_non_text(swt, labels, components)

    labels = labels.astype(np.float32) / labels.max()
    l = (labels * 255.).astype(np.uint8)

    l = cv.cvtColor(l, cv.COLOR_GRAY2RGB)
    l = cv.LUT(l, build_colormap())
    cv.imshow('components', l)

    swt = (255 * swt / swt.max()).astype(np.uint8)
    cv.imshow('swt', swt)

    # START HERE
    result_r = l.copy()
    kernel = np.ones((9, 9), np.uint8)
    opening = cv.morphologyEx(result_r, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    # Convert the image to gray scale
    gray_r = cv.cvtColor(closing, cv.COLOR_BGR2GRAY)

    # Performing OTSU threshold
    ret, thresh1 = cv.threshold(gray_r, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)
    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    # Appplying dilation on the threshold image
    dilation = cv.dilate(thresh1, rect_kernel, iterations=1)

    cv.imshow("gray_r", gray_r)
    cv.imshow("dilation", dilation)

    # Finding contours
    contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    result_r2 = l.copy()

    white_board = np.full(result_r2.shape[:2], 255, np.uint8)

    # Looping through the identified contours
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        area = w * h

        if (area > 100 and area < 1000):
            # Drawing a rectangle on copied image
            rect = cv.rectangle(white_board, (x, y), (x + w, y + h), (0, 0, 255), -1)
        # rect = cv.rectangle(result_r2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow("sdsdf", white_board)
    # hsv = cv2.cvtColor(white_board, cv2.COLOR_BGR2HSV)

    # hist = cv.calcHist([white_board], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # cv.imshow("hist", hist)
