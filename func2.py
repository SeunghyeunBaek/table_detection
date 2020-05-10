## req
import cv2
import numpy as np
import pytesseract
import pandas as pd

## functions

def imshow(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# line
def get_line(img, erod_size, min_line_size):

    if min_line_size % 2 == 0:
        min_line_size += 1

    erod_ker = np.ones((erod_size, erod_size), np.uint8)
    dil_row_ker = np.ones((1, min_line_size), np.uint8)
    dil_col_ker = np.ones((min_line_size, 1), np.uint8)

    erod_img = cv2.erode(img, erod_ker, iterations=1)
    row_img = cv2.bitwise_not(cv2.dilate(erod_img, dil_row_ker, iterations=1))
    col_img = cv2.bitwise_not(cv2.dilate(erod_img, dil_col_ker, iterations=1))
    
    res_dict = {
        'row_img': row_img
        , 'col_img': col_img
    }
    
    return res_dict


def get_box(img, back_img, line_color, line_width, min_line_size, type_):

    cnts, _ = cv2.findContours(img,  cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mar = min_line_size // 2
    coord_ls = list()

    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if type_ == 'row':
            mod_coord = ((x - mar, y), (x + w + mar - 1, y + h + - 1))
            if w > min_line_size:
                cv2.rectangle(back_img, mod_coord[0], mod_coord[1], line_color, line_width)
                coord_ls.append(y)  
                
        elif type_ == 'col':
            mod_coord = ((x, y - mar), (x + w - 1, y + h + 14))
            if h > min_line_size:
                cv2.rectangle(back_img, mod_coord[0], mod_coord[1], line_color, line_width)
                coord_ls.append(x)
        else:
            mod_coord = ((x, y) , (x+w, y+h))
            if min_line_size < h < img.shape[0] - 10:
                cv2.rectangle(back_img, mod_coord[0], mod_coord[1], line_color, line_width)
                coord_ls.append(mod_coord)

    res = {
        'img': back_img
        , 'coord': coord_ls
    }

    return res

def get_text(img, erod_size, iter):
    erod_ker = np.ones((erod_size, erod_size), np.uint8)
    text_area = cv2.erode(img, erod_ker, iterations=iter)
    return text_area
    

def remove_noise(img, min_line_size):
    ker_area = np.square(min_line_size)  # 커널 크기
    noi_ker = np.ones((min_line_size, min_line_size), np.float32) / ker_area  # 커널
    mean_pix_val_img = cv2.filter2D(img, -1, noi_ker)  # 커널 밝기 평균값
    
    n_bk_line = 1
    thr = (255 * min_line_size * (min_line_size - n_bk_line)) / ker_area + 1
    _, noi_img = cv2.threshold(mean_pix_val_img, thr , 255, cv2.THRESH_BINARY)    
    
    noi_img = cv2.bitwise_not(noi_img)
    img_ = cv2.add(img, noi_img)
    img_ = cv2.bitwise_not(img_)
    rm_noi_img = cv2.add(img, img_)
    
    return rm_noi_img


def get_mod_line(line_ls, line_mar):

    mod_line_ls = list()
    for n, val in enumerate(line_ls):
        if n == len(line_ls) - 1:
            mod_line_ls.append(val)
        else:
            diff = line_ls[n+1] - line_ls[n]
            if diff > line_mar:
                mod_line_ls.append(val)
    return mod_line_ls


def get_cell_coord(x_ls, y_ls):
    pix_coord_ls = list()
    cell_coord_ls = list()
    for n_y, y in enumerate(y_ls):
        if n_y == len(y_ls) - 1:
            pass
        else:
            for n_x, x in enumerate(x_ls):
                if n_x == len(x_ls) - 1:
                    pass
                else:
                    pix_coord= ((x, y), (x_ls[n_x + 1], y_ls[n_y + 1]))
                    pix_coord_ls.append(pix_coord)
                    cell_coord_ls.append((n_y, n_x))
    res = {
        'pix_coord': pix_coord_ls
        , 'mat_coord': cell_coord_ls
    }
    return res


def map_cell_text(cell_pix_mat_coord_ls, text_coord_ls):
    cell_text_coord_ls = list()
    for text_coord in text_coord_ls:
        tx1, ty1 = text_coord[0]
        tx2, ty2 = text_coord[1]
        tx_mid, ty_mid = (tx1+tx2)/2, (ty1+ty2)/2

        for cell_coord in cell_pix_mat_coord_ls:
            cell_pix_coord, cell_mat_coord = cell_coord
            cpx1, cpy1 = cell_pix_coord[0]
            cpx2, cpy2 = cell_pix_coord[1]

            if (cpx1 < tx_mid < cpx2) & (cpy1 < ty_mid < cpy2):

                cell_text_coord_ls.append((cell_pix_coord, cell_mat_coord, text_coord))   

    return cell_text_coord_ls

        
def rec_text(img, cell_text_coord_ls, filename, path, conf):
    cell_text_dict = { mat: [] for pix, mat, text in cell_text_coord_ls}

    for cell_text_coord in cell_text_coord_ls:
        cell_mat_coord = cell_text_coord[1]
        tx1, ty1 = cell_text_coord[2][0]
        tx2, ty2 = cell_text_coord[2][1]
        tx_mid, ty_mid = (tx1 + tx2)/2, (ty1 + ty2)/2
        crop_img = img[ty1:ty2, tx1:tx2]
        
        text = pytesseract.image_to_string(crop_img, lang='kor', config=conf)
        
        cell_text_dict[cell_mat_coord].append((text, (tx_mid, ty_mid)))
        
        sp_filename = path + filename.split('.')[0] + '_' + str(cell_mat_coord) + '_' + str((tx1, ty1, tx2, ty2)) + '.' + filename.split('.')[1]
        
        cv2.imwrite(sp_filename, crop_img)
    return cell_text_dict

        
def result_to_csv(n_row, n_col, cell_text_dict, filename):
    
    table_arr = np.chararray(shape=(n_row, n_col), itemsize=30, unicode=True)
    
    for cell_mat_coord, text_ls in cell_text_dict.items():
        row = cell_mat_coord[0]
        col = cell_mat_coord[1]
        if len(text_ls) == 1:
            pass
        else:
            sorted_text_ls = sorted(text_ls, key=lambda x: x[1][0])  # x좌표정렬
            for i in range(len(sorted_text_ls)):
                if sorted_text_ls[i][1][1] - sorted_text_ls[0][1][1] > 20:
                    pop = sorted_text_ls.pop(i)
                    sorted_text_ls.append(pop)
            cell_text_dict[cell_mat_coord] = sorted_text_ls
        text = '  '.join([str_ for str_, _ in cell_text_dict[cell_mat_coord]])
        table_arr[row, col] = text
            
    df = pd.DataFrame(table_arr)
    
    return df
    