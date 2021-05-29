import os

import cv2 as cv2

from envelop_recogniser.constants import threshold_stroke_width_percentage, \
    threshold_connected_component_position_count, threshold_line_height_to_word_gap_ratio_for_handwritten_letters, \
    iteration_count, threshold_rectangle_height_width_ratio

RECT = {'x1': int, 'y1': int, 'x2': int, 'y2': int, 'end_point': (int, int),
        'start_point': (int, int), 'w': int, 'h': int, 'length': int, 'area': int,
        'distance_bottom_next': int, 'distance_right_next': int, 'label': int, 'not_consumed': bool}


def save_output(image_name, image, prefix='ci'):
    folder_name = str(threshold_stroke_width_percentage) + "_" + str(
        threshold_connected_component_position_count) + "_" + str(
        threshold_line_height_to_word_gap_ratio_for_handwritten_letters) + "_" + str(
        threshold_rectangle_height_width_ratio) + "_" + "output" + "_" + str(iteration_count)
    # name_only = os.path.splitext(image_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    filename = folder_name + "/" + prefix + "_" + image_name
    if not cv2.imwrite(filename, image):
        raise Exception("Could not write image")


def generate_selected_rectangles(dilation) -> [[], [], []]:
    rect_heights = []
    rect_widths = []
    selected_rectangles = []
    contours_r, hierarchy_r = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_r:
        x1, y1, w, h = cv2.boundingRect(cnt)
        rect = {'x1': x1, 'y1': y1, 'x2': x1 + w, 'y2': y1 + h, 'end_point': (x1 + w, y1 + h),
                'start_point': (x1, y1), 'w': w, 'h': h, 'length': -1, 'area': w * h,
                'distance_bottom_next': None, 'distance_right_next': None, 'label': -1, 'label_2': -1, 'label_3': -1,
                'label_4': -1, 'not_consumed': True}
        rect_heights.append(h)
        rect_widths.append(w)
        selected_rectangles.append(rect)
    return selected_rectangles, rect_heights, rect_widths


def rename_labels(old_name, new_name, rectangles):
    for r in rectangles:
        if r['label'] == old_name:
            r['label'] = new_name


def print_label(white_board, data, label_name, color=(255, 0, 255)):
    if data:
        cX = int((data['start_point'][0] + data['end_point'][0]) / 2) - 30
        cY = int((data['start_point'][1] + data['end_point'][1]) / 2) - 30

        d_y = 30
        label_names = label_name.split(' ')
        for label in label_names:
            cY = cY + d_y
            cv2.putText(white_board, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)


def label_rects(rects: [], half_width_cutoff, half_height_cutoff, prop='label') -> []:
    selected_rects: [] = []
    rects.sort(key=lambda r: r['x1'])
    label = 0
    index = 0

    for selected_rect in rects:
        exists = False
        stored_label = -1
        if selected_rect[prop] > -1:
            stored_label = selected_rect[prop]
            exists = True
        else:
            stored_label = label
            selected_rect[prop] = stored_label

        rects_copy = rects[index + 1:].copy()
        index2 = 0
        for rc in rects_copy:
            y2y1 = abs(selected_rect['y2'] - rc['y1']) < half_height_cutoff
            y1y2 = abs(selected_rect['y1'] - rc['y2']) < half_height_cutoff
            y1y1 = abs(selected_rect['y1'] - rc['y1']) < half_height_cutoff
            y2y2 = abs(selected_rect['y2'] - rc['y2']) < half_height_cutoff
            y1y1y2_srs = selected_rect['y1'] <= rc['y1'] <= selected_rect['y2']
            y1y2y2_srs = selected_rect['y1'] <= rc['y2'] <= selected_rect['y2']

            y1y1y2_rsr = rc['y1'] <= selected_rect['y1'] <= rc['y2']
            y1y2y2_rsr = rc['y1'] <= selected_rect['y2'] <= rc['y2']

            x1x1 = abs(selected_rect['x1'] - rc['x1']) < half_width_cutoff
            x1x2 = abs(selected_rect['x1'] - rc['x2']) < half_width_cutoff
            x2x1 = abs(selected_rect['x2'] - rc['x1']) < half_width_cutoff
            x2x2 = abs(selected_rect['x2'] - rc['x2']) < half_width_cutoff
            x1x1x2_srs = selected_rect['x1'] <= rc['x1'] <= selected_rect['x2']
            x1x2x1_srs = selected_rect['x1'] <= rc['x2'] <= selected_rect['x2']
            x1x1x2_rsr = rc['x1'] <= selected_rect['x1'] <= rc['x2']
            x1x2x2_rsr = rc['x1'] <= selected_rect['x2'] <= rc['x2']

            y_portion = y1y2 or y2y2 or y2y1 or y1y1 or y1y1y2_srs or y1y2y2_srs or y1y1y2_rsr or y1y2y2_rsr
            x_portion = x1x1 or x1x2 or x2x1 or x2x2 or x1x1x2_srs or x1x2x1_srs or x1x1x2_rsr or x1x2x2_rsr

            if y_portion and x_portion:
                if rects[index2 + 1 + index][prop] > -1:
                    rename_labels(rects[index2 + 1 + index][prop], stored_label, rects)
                else:
                    rects[index2 + 1 + index][prop] = stored_label
                    selected_rects.append(rects[index2 + 1 + index])
            index2 += 1

        if not exists:
            selected_rects.append(selected_rect)
        index += 1
        label += 1
    return selected_rects


def merge_rectangles_2(rects) -> {'x1': int, 'y1': int, 'x2': int, 'y2': int, 'end_point': (int, int),
                                  'start_point': (int, int), 'length': int}:
    rects.sort(key=lambda r: r['x1'], reverse=False)
    x_min = rects[0]['x1']
    rects.sort(key=lambda r: r['x2'], reverse=True)
    x_max = rects[0]['x2']
    rects.sort(key=lambda r: r['y2'], reverse=True)
    y_max = rects[0]['y2']
    rects.sort(key=lambda r: r['y1'], reverse=False)
    y_min = rects[0]['y1']

    return {'x1': x_min, 'y1': y_min, 'x2': x_max, 'y2': y_max, 'end_point': (x_max, y_max), 'w': x_max - x_min,
            'h': y_max - y_min,
            'area': (x_max - x_min) * (y_max - y_min),
            'start_point': (x_min, y_min), 'label': rects[0]['label'], 'label_2': rects[0]['label_2'],
            'label_3': rects[0]['label_3']}


def purify_x_rects(selected_rect: RECT, width_r: int, width_cutoff: int) -> []:
    x_purified_rects: [] = []
    for i in range(width_cutoff, width_r + 1, width_cutoff):
        if selected_rect['x1'] < i:
            if selected_rect['x2'] <= i:
                # print("width within 1 cell")
                x_purified_rects.append(selected_rect)
                return x_purified_rects
            elif selected_rect['x2'] <= i + width_cutoff:
                # print("cut rect vertically. out is 2 rects")
                rect1 = selected_rect.copy()
                rect1['x2'] = i
                rect1['end_point'] = (i, rect1['y2'])
                rect1['area'] = abs((rect1['x2'] - rect1['x1']) * (rect1['y2'] - rect1['y1']))
                x_purified_rects.append(rect1)
                rect2 = selected_rect.copy()
                rect2['x1'] = i
                rect2['start_point'] = (i, rect2['y1'])
                rect2['area'] = abs((rect2['x2'] - rect2['x1']) * (rect2['y2'] - rect2['y1']))
                x_purified_rects.append(rect2)
                return x_purified_rects
            else:
                print("this rect is too long. neglect this")
                return x_purified_rects
        else:
            continue
    print("empty rect")
    return x_purified_rects


def purify_y_rects(selected_rect: RECT, height_r: int, height_cutoff: int) -> []:
    y_purified_rects: [] = []
    for j in range(height_cutoff, height_r + 1, height_cutoff):
        if selected_rect['y1'] < j:
            if selected_rect['y2'] <= j:
                # print("width within 1 cell")
                y_purified_rects.append(selected_rect)
                return y_purified_rects
            elif selected_rect['y2'] <= height_cutoff + j:
                # print("cut rect vertically. out is 2 rects")
                rect1 = selected_rect.copy()
                rect1['y2'] = j
                rect1['end_point'] = (rect1['x2'], j)
                rect1['area'] = abs((rect1['x2'] - rect1['x1']) * (rect1['y2'] - rect1['y1']))
                y_purified_rects.append(rect1)
                rect2 = selected_rect.copy()
                rect2['y1'] = j
                rect2['start_point'] = (rect2['x1'], j)
                rect2['area'] = abs((rect2['x2'] - rect2['x1']) * (rect2['y2'] - rect2['y1']))
                y_purified_rects.append(rect2)
                return y_purified_rects
            else:
                print("this rect is too long. neglect this")
                return y_purified_rects
        else:
            continue
    print("empty rect")
    return y_purified_rects


def evaluate_cells(cell_extras):
    total_rect_count = 0
    total_rect_area = 0

    for cell_extra in cell_extras:
        total_rect_count = len(cell_extra['rects']) + total_rect_count
        for r in cell_extra['rects']:
            total_rect_area = total_rect_area + r['area']

    address_probability_array = []
    for cell_extra in cell_extras:
        count_probability = len(cell_extra['rects']) / total_rect_count
        total_cell_rects_area = 0
        for r in cell_extra['rects']:
            total_cell_rects_area = total_cell_rects_area + r['area']
        area_probability = total_cell_rects_area / total_rect_area
        probability = count_probability * area_probability
        address_probability_array.append(
            {'val': cell_extra, 'probability': probability,
             'p_string': '%.5f %.5f %.5f' % (count_probability, area_probability, probability)})

    address_probability_array.sort(key=lambda r: (r['probability']), reverse=True)
    return address_probability_array
