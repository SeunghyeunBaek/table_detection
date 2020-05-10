## req
import cv2
import copy

## functions

def imshow(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# line

def remove_line(img, row_img, col_img):
    
    rm_line_img = copy.deepcopy(img)
    
    for row_id in range(row_img.shape[0]):
        for col_id in range(row_img.shape[1]):
            if row_img[row_id, col_id] == 0:
                rm_line_img[row_id, col_id-100: col_id + 100] = 255
            if col_img[row_id, col_id] == 0:
                rm_line_img[row_id-100: col_id+100, col_id] = 255
                
    return rm_line_img
                
def get_box(img, type, back_img, margin=0, min_height=20, min_width=20, line_color=(255, 0, 0), line_width=2):
    
#     if back_img != cv2.threshold(img, 255, 255, cv2.THRESH_BINARY):
#         back_img = cv2.
    contour, hier = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    box_coord_ls = list()
    
    for cnt_id, cnt in enumerate(contour):
        x, y, w, h = cv2.boundingRect(cnt)
        if type=='row':
            if (min_width < w < back_img.shape[1] - 10):
                cv2.rectangle(back_img, (x, y), (x+w, y+h), line_color, line_width)
                box_coord_ls.append((x, y, x+w, y+h))
        elif type=='col':
            if (min_height < h < back_img.shape[0] - 10):
                cv2.rectangle(back_img, (x, y), (x+w, y+h), line_color, line_width)
                box_coord_ls.append((x, y, x+w, y+h))
        elif type=='box':
            if (min_height < h):
                cv2.rectangle(back_img, (x, y), (x+w, y+h), line_color, line_width)
                box_coord_ls.append((x, y, x+w, y+h))
        else:
            pass
    
    res_dict = {
        'img': back_img
        , 'coord': box_coord_ls
    }

    return res_dict

def draw_edge(img, margin=1):
    img[0:margin, :], img[-margin:, :], img[:, 0:margin], img[:, -margin:] = 255, 255, 255, 255
    return img


def get_cell(img, row_img, col_img, min_size=30):
    

    back_img = copy.deepcopy(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))

    # 이미지 전체 가장자리 표시
    row_img, col_img = draw_edge(row_img), draw_edge(col_img)

    # 윤곽선 검출
    row_contour, _ = cv2.findContours(row_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    col_contour, _ = cv2.findContours(col_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 라인 좌표 초기화
    xline_ls , yline_ls= list(), list()

    img_w, img_h = img.shape[1], img.shape[0]

    # 라인 좌표 구하기
    for row_id, row_cnt in enumerate(row_contour):
        x, y, w, h = cv2.boundingRect(row_cnt)
        if w > min_size:
            yline_ls.append(y)
            cv2.rectangle(back_img, (0, y), (img_w, y), (255, 0, 0), 2)

    for col_id, col_cnt in enumerate(col_contour):
        x, y, w, h = cv2.boundingRect(col_cnt)
        xline_ls.append(x)
        cv2.rectangle(back_img, (x, 0), (x, img_h), (255, 0, 0), 2)

    # 정렬
    xline_ls, yline_ls = sorted(xline_ls), sorted(yline_ls)

    # cell 좌표 구하기
    n_row, n_col = len(yline_ls), len(xline_ls)
    n_cell = (n_row - 1) * (n_col - 1)
    cell_id_ls, cell_coord_ls = [-1 for i in range(n_cell)], [-1 for i in range(n_cell)]

    for n_y, y in enumerate(yline_ls):
        # for all rows
        if n_y == n_row -1:
            pass
        else:
            for n_x, x in enumerate(xline_ls):
                if n_x == n_col - 1:
                    pass
                else:
                    cur_loop_id = (n_col-1) * n_y + n_x
                    # print(f'{n_col - 1}| {n_y} {n_x} | {cur_loop_id}')
                    cell_id_ls[cur_loop_id] = (n_y, n_x)
                    cell_coord_ls[cur_loop_id] = (x, y)

    res_dict = {
        'img': back_img
        , 'line_coord_x': xline_ls
        , 'line_coord_y': yline_ls
        , 'cell_id': cell_id_ls
        , 'cell_coord': cell_coord_ls
    }

    
    return res_dict



    
        
        
        