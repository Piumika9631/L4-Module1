import statistics
from typing import TypeVar, NamedTuple, List, Optional, Tuple
import cv2
import numpy as np
from scipy.spatial import ConvexHull

from envelop_recogniser.constants import threshold_stroke_width_percentage, \
    threshold_connected_component_position_count, threshold_rectangle_height_width_ratio
from envelop_recogniser.id_colors import build_colormap
from envelop_recogniser.identify_elements.skew_correction import rotate_bound
from envelop_recogniser.utils import generate_selected_rectangles, label_rects, print_label, merge_rectangles_2, \
    purify_x_rects, purify_y_rects, evaluate_cells

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
    edges = cv2.Canny(im, lo, hi, apertureSize=window)
    # Note that the output is either 255 for edges or 0 for other pixels.
    # Conversion to float wastes space, but makes the return value consistent
    # with the other methods.
    return edges.astype(np.float32) / 255.


def get_gradients(im: Image) -> Gradients:
    # Obtains the image gradients by means of a 3x3 Scharr filter.
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
    rays_temp = []
    rays_map = []

    # Find a pixel that lies on an edge.
    height, width = im.shape[0:2]
    for y in range(height):
        for x in range(width):
            # Edges are either 0. or 1.
            if edges[y, x] < .5:
                continue
            # Calculate SWT
            ray, stroke_width = swt_process_pixel(Position(x=x, y=y), edges, directions, out=swt,
                                                  dark_on_bright=dark_on_bright)

            if ray:
                rays_map.append({'ray': ray, 'stroke_width': stroke_width, 'width': str(int(stroke_width))})
                rays_temp.append(ray)

    total_rays_count = len(rays_map)
    ray_count_map = {}
    ray_percent_count_map = {}
    for rayData in rays_map:
        if rayData['width'] in ray_count_map:
            ray_count_map[rayData['width']] += 1
        else:
            ray_count_map[rayData['width']] = 1

    for key in ray_count_map:
        value = ray_count_map[key]
        percentage = value / total_rays_count * 100
        ray_percent_count_map[key] = int(percentage)
        # remove outliers like strokes over stamps/labels
        if int(percentage) > threshold_stroke_width_percentage:
            for rayData in rays_map:
                if rayData['width'] == key:
                    ray = rayData['ray']
                    o = {'ray': ray, 'stroke_width': rayData['stroke_width']}
                    rays.append(o)

    # Plot rays into an image
    for rayObject in rays:
        ray = rayObject['ray']
        stroke_width_a = rayObject['stroke_width']
        for p in ray:
            swt[p.y, p.x] = min(stroke_width_a, swt[p.y, p.x])

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
            return None, 0.0
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
            return None, 0.0
        # Paint each of the pixels on the ray with their determined stroke width
        stroke_width = np.sqrt((cur_x - pos.x) * (cur_x - pos.x) + (cur_y - pos.y) * (cur_y - pos.y))

        return ray, stroke_width

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
            if len(component) > threshold_connected_component_position_count:
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_r = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    edges = cv2.Canny(thresh_r, 50, 200)
    lines = cv2.HoughLinesP(edges, rho=1, theta=1 * np.pi / 180,
                            threshold=10, minLineLength=20, maxLineGap=3)
    # Draw lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv2.imshow('result', image)


def merge_rectangles(rects) -> {'x1': int, 'y1': int, 'x2': int, 'y2': int, 'end_point': (int, int),
                                'start_point': (int, int), 'length': int}:
    x_max = 0
    x_min = 100000
    y_max = 0
    y_min = 100000
    for position in rects:
        x1 = position[0]
        y1 = position[1]
        if x_max < x1:
            x_max = x1
        if x_min > x1:
            x_min = x1
        if y_max < y1:
            y_max = y1
        if y_min > y1:
            y_min = y1

    length = int(np.sqrt((x_min - x_max) ** 2 + (y_min - y_max) ** 2))
    return {'x1': x_min, 'y1': y_min, 'x2': x_max, 'y2': y_max, 'end_point': (x_max, y_max),
            'area': (x_max - x_min) * (y_max - y_min),
            'start_point': (x_min, y_min), 'length': length}


def swt_main(image):
    cropped_envelop_image = image
    gray_cropped_envelop = open_grayscale(image)

    # Find the edges in the image and the gradients.
    edges = get_edges(gray_cropped_envelop)
    # Find image gradients
    gradients = get_gradients(gray_cropped_envelop)

    # Apply the Stroke Width Transformation and get character identified image.
    swt = apply_swt(gray_cropped_envelop, edges, gradients)

    # Apply Connected Components labelling to get words
    labels, components = connected_components(swt)
    white_board = np.full(cropped_envelop_image.shape[:3], (255, 255, 255), np.uint8)

    component_rects = []
    for comp in components:
        rect = merge_rectangles(comp)
        component_rects.append(rect)
        white_board = cv2.rectangle(white_board, rect['start_point'], rect['end_point'], (0, 0, 0), -1)

    labels = labels.astype(np.float32) / labels.max()
    l = (labels * 255.).astype(np.uint8)

    l = cv2.cvtColor(l, cv2.COLOR_GRAY2RGB)
    l = cv2.LUT(l, build_colormap())
    # cv.imshow('components', l)

    swt = (255 * swt / swt.max()).astype(np.uint8)
    # cv.imshow('swt', swt)

    # Analyse the rectangles (identified word blobs) of result image
    result_image = white_board.copy()
    kernel = np.ones((9, 9), np.uint8)
    opening = cv2.morphologyEx(result_image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    gray_r = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)

    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray_r, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, dilate_kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    horizontal_rectangles = []
    selected_rectangles = []
    vertical_rectangles = []
    rect_heights = []
    rect_widths = []
    rotation_angle = -1  # default value
    for cnt in contours:
        x1, y1, w, h = cv2.boundingRect(cnt)
        rect = {'x1': x1, 'y1': y1, 'x2': x1 + w, 'y2': y1 + h, 'end_point': (x1 + w, y1 + h),
                'start_point': (x1, y1), 'w': w, 'h': h, 'length': -1, 'area': w * h, 'distance_bottom_next': None,
                'distance_right_next': None, 'label': -1, 'label_2': -1, 'label_3': -1, 'label_4': -1,
                'not_consumed': True}
        rect_heights.append(h)
        rect_widths.append(w)
        selected_rectangles.append(rect)
        cv2.rectangle(white_board, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), -1)
        if w / h > threshold_rectangle_height_width_ratio:
            horizontal_rectangles.append(rect)

        if h / w > threshold_rectangle_height_width_ratio:
            vertical_rectangles.append(rect)

    # calculate rotation angle
    if len(horizontal_rectangles) < len(vertical_rectangles):
        rotation_angle = 90
    elif len(horizontal_rectangles) > len(vertical_rectangles):
        rotation_angle = 0
    else:
        # h > w : calculate the angle using width and height
        if white_board.shape[0] > white_board.shape[1]:
            rotation_angle = 90
        else:
            rotation_angle = 0

    print("rotation_angle: " + str(rotation_angle))
    cropped_image = cropped_envelop_image.copy()

    if rotation_angle == 90:
        dilation = rotate_bound(dilation, rotation_angle)
        white_board = rotate_bound(white_board, rotation_angle)
        cropped_image = rotate_bound(cropped_envelop_image, rotation_angle)
        # need to get rects again
        selected_rectangles, rect_heights, rect_widths = generate_selected_rectangles(dilation)

    height_r = white_board.shape[0]
    width_r = white_board.shape[1]

    # start vertical divide image
    width_cutoff = int(width_r / 3)
    height_cutoff = int(height_r / 3)

    median_rect_h = statistics.median(rect_heights)
    median_rect_w = statistics.median(rect_widths)
    mean_rect_h = statistics.mean(rect_heights)  # average line height
    mean_rect_w = statistics.mean(rect_widths)  # average word width

    print('median_rect_h %.2f, mean_rect_h %.2f' % (median_rect_h, mean_rect_h))
    print('median_rect_w %.2f, mean_rect_w %.2f' % (median_rect_w, mean_rect_w))

    # Line spacing is the space between each line in a paragraph.
    # Word allows you to customize the line spacing to be single spaced (one line high),
    # double spaced (two lines high), or any other amount you want.
    # The default spacing in Word is 1.08 lines, which is slightly larger than single spaced.
    additional_spacing_h_factor = mean_rect_h
    additional_spacing_w_factor = mean_rect_h * 1.5

    half_width_cutoff = additional_spacing_w_factor
    half_height_cutoff = additional_spacing_h_factor
    diagonal_cutoff = np.sqrt(half_height_cutoff ** 2 + half_width_cutoff ** 2)

    selected_rectangles.sort(key=lambda rec: rec['x1'])

    current_index = 0
    rect_count = len(selected_rectangles)
    labeled_rects = label_rects(selected_rectangles, half_width_cutoff * 1.0, half_height_cutoff)

    cluster_dict = dict()
    for r in labeled_rects:
        label = r['label']
        print_label(white_board, r, str(r['label']), (0, 0, 200))
        if label in cluster_dict:
            temp = cluster_dict[label]
            temp.append(r)
            cluster_dict[label] = temp
        else:
            cluster_dict[label] = [r]

    cluster_as_rects = []
    for cluster_name in cluster_dict:
        cluster = cluster_dict[cluster_name]
        rect = merge_rectangles_2(cluster)
        cluster_as_rects.append(rect)

    labeled_rects = label_rects(cluster_as_rects, half_width_cutoff * 1.0, half_height_cutoff, 'label_2')

    cluster_dict = dict()
    for r in labeled_rects:
        label = r['label_2']
        print_label(white_board, r, str(r['label_2']), (255, 0, 200))
        if label in cluster_dict:
            temp = cluster_dict[label]
            temp.append(r)
            cluster_dict[label] = temp
        else:
            cluster_dict[label] = [r]

    cluster_as_rects = []
    for cluster_name in cluster_dict:
        cluster = cluster_dict[cluster_name]
        rect = merge_rectangles_2(cluster)
        cluster_as_rects.append(rect)

    labeled_rects = label_rects(cluster_as_rects, half_width_cutoff * 1.0, half_height_cutoff, 'label_3')

    cluster_dict = dict()
    for r in labeled_rects:
        label = r['label_3']
        print_label(cropped_image, r, str(r['label_3']), (200, 0, 0))
        if label in cluster_dict:
            temp = cluster_dict[label]
            temp.append(r)
            cluster_dict[label] = temp
        else:
            cluster_dict[label] = [r]

    cluster_as_rects = []
    for cluster_name in cluster_dict:
        cluster = cluster_dict[cluster_name]
        rect = merge_rectangles_2(cluster)
        cluster_as_rects.append(rect)
        white_board = cv2.rectangle(white_board, rect['start_point'], rect['end_point'], (255, 0, 255), 1)
        cropped_image = cv2.rectangle(cropped_image, rect['start_point'], rect['end_point'], (255, 0, 255), 1)

    # cv2.imshow("temp", cropped_image)
    # print(resultr)
    cluster_as_rects_2 = []
    # for large_rect in cluster_as_rects:
    #     rect = merge_rectangles(large_rect)
    #     cluster_as_rects_2.append(rect)
    #     white_board = cv2.rectangle(white_board, rect['start_point'], rect['end_point'], (255, 0, 255), 1)

    # for selected_rect in selected_rectangles:
    #     print(selected_rect)
    #     if current_index + 1 == rect_count:
    #         break
    #     next_rect = selected_rectangles[current_index + 1]
    #     selected_rect['distance_right_next'] = abs(next_rect['x1'] - selected_rect['x2'])
    #     current_index = current_index + 1
    #
    # selected_rectangles.sort(key=lambda r: r['y1'])
    # current_index = 0
    # for selected_rect in selected_rectangles:
    #     print(selected_rect)
    #     if current_index + 1 == rect_count:
    #         break
    #     next_rect = selected_rectangles[current_index + 1]
    #     selected_rect['distance_bottom_next'] = abs(next_rect['y1'] - selected_rect['y2'])
    #     current_index = current_index + 1

    # selected_rectangles = cluster_as_rects
    x_purified_rects = []

    for selected_rect in selected_rectangles:
        new_rects = purify_x_rects(selected_rect, width_r, width_cutoff)
        x_purified_rects.extend(new_rects)

    purified_rects = []
    for x_rect in x_purified_rects:
        new_rects = purify_y_rects(x_rect, height_r, height_cutoff)
        purified_rects.extend(new_rects)

    c_1_limit = width_cutoff
    c_2_limit = width_cutoff + width_cutoff
    c_3_limit = width_cutoff + width_cutoff + width_cutoff

    r_1_limit = height_cutoff
    r_2_limit = height_cutoff + height_cutoff
    r_3_limit = height_cutoff + height_cutoff + height_cutoff

    cell_1_1 = []  # top left
    cell_1_2 = []  # top middle
    cell_1_3 = []  # top right

    cell_2_1 = []  # middle left
    cell_2_2 = []  # middle middle
    cell_2_3 = []  # middle right

    cell_3_1 = []  # bottom left
    cell_3_2 = []  # bottom middle
    cell_3_3 = []  # bottom right

    # all_rect_cords = []
    # total_rect_area = 0

    for selected_rect in purified_rects:
        # for selected_rect in selected_rectangles:
        x1 = selected_rect['x1']
        y1 = selected_rect['y1']
        x2 = selected_rect['x2']
        y2 = selected_rect['y2']
        # all_rect_cords.append([x1, y1, x2, y2])
        selected_rect['ne_distance'] = -1
        selected_rect['se_distance'] = -1
        selected_rect['sw_distance'] = -1
        selected_rect['nw_distance'] = -1
        # rect_heights.append(abs(x1 - x2))
        # rect_widths.append(abs(y1 - y2))
        # total_rect_area += selected_rect['area']
        cv2.rectangle(white_board, (x1, y1), selected_rect['end_point'], (0, 0, 255), 2)
        if 0 <= x1 < c_1_limit and 0 <= y1 < r_1_limit:
            cell_1_1.append(selected_rect)
        elif c_1_limit <= x1 < c_2_limit and 0 <= y1 < r_1_limit:
            cell_1_2.append(selected_rect)
        elif c_2_limit <= x1 and 0 <= y1 < r_1_limit:
            cell_1_3.append(selected_rect)
        elif 0 <= x1 < c_1_limit and r_1_limit <= y1 < r_2_limit:
            cell_2_1.append(selected_rect)
        elif c_1_limit <= x1 < c_2_limit and r_1_limit <= y1 < r_2_limit:
            cell_2_2.append(selected_rect)
        elif c_2_limit <= x1 and r_1_limit <= y1 < r_2_limit:
            cell_2_3.append(selected_rect)
        elif 0 <= x1 < c_1_limit and r_2_limit <= y1:
            cell_3_1.append(selected_rect)
        elif c_1_limit <= x1 < c_2_limit and r_2_limit <= y1:
            cell_3_2.append(selected_rect)
        elif c_2_limit <= x1 and r_2_limit <= y1:
            cell_3_3.append(selected_rect)

    cell_1_1_with_extra = {'name': 'A', 'rects': cell_1_1, 'start_point': (0, 0), 'end_point': (c_1_limit, r_1_limit)}
    cell_1_2_with_extra = {'name': 'B', 'rects': cell_1_2, 'start_point': (c_1_limit, 0),
                           'end_point': (c_2_limit, r_1_limit)}
    cell_1_3_with_extra = {'name': 'C', 'rects': cell_1_3, 'start_point': (c_2_limit, 0),
                           'end_point': (c_3_limit, r_1_limit)}
    cell_2_1_with_extra = {'name': 'D', 'rects': cell_2_1, 'start_point': (0, r_1_limit),
                           'end_point': (c_1_limit, r_2_limit)}
    cell_2_2_with_extra = {'name': 'E', 'rects': cell_2_2, 'start_point': (c_1_limit, r_1_limit),
                           'end_point': (c_2_limit, r_2_limit)}
    cell_2_3_with_extra = {'name': 'F', 'rects': cell_2_3, 'start_point': (c_2_limit, r_1_limit),
                           'end_point': (c_3_limit, r_2_limit)}
    cell_3_1_with_extra = {'name': 'G', 'rects': cell_3_1, 'start_point': (0, r_2_limit),
                           'end_point': (c_1_limit, r_3_limit)}
    cell_3_2_with_extra = {'name': 'H', 'rects': cell_3_2, 'start_point': (c_1_limit, r_2_limit),
                           'end_point': (c_2_limit, r_3_limit)}
    cell_3_3_with_extra = {'name': 'I', 'rects': cell_3_3, 'start_point': (c_2_limit, r_2_limit),
                           'end_point': (c_3_limit, r_3_limit)}

    cells = [cell_1_1_with_extra, cell_1_2_with_extra, cell_1_3_with_extra,
             cell_2_1_with_extra, cell_2_2_with_extra, cell_2_3_with_extra,
             cell_3_1_with_extra, cell_3_2_with_extra, cell_3_3_with_extra]

    address_probability_array = evaluate_cells(cells)

    for cell_ex in cells:
        start_point_rr = cell_ex['start_point']
        end_point_rr = cell_ex['end_point']
        cropped_image = cv2.rectangle(cropped_image, start_point_rr, end_point_rr, (0, 100, 0), 1)
        # white_board_manual = cv2.rectangle(white_board_manual, start_point_rr, end_point_rr, (25, 0, 255), 1)
        white_board = cv2.rectangle(white_board, start_point_rr, end_point_rr, (25, 0, 255), 1)

    color_var = 25
    r_c = 255
    g_c = 0
    b_c = 255
    for data in address_probability_array:
        r_c -= color_var
        g_c += color_var
        color = (b_c, g_c, r_c)
        print_label(white_board, data['val'], data['p_string'],
                    color)

    vis = np.concatenate((white_board, cropped_image), axis=0)

    # cv2.imshow('white board', white_board)
    # cv2.imshow('cropped image', cropped_image)
    # cv2.imshow('vis', vis)

    return white_board, cropped_image, vis
