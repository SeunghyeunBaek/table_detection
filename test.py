from func import *
import cv2
import numpy as np

path = 'img/'            # 원본이미지 경로
proc_path = 'proc_img/'  # 이진화 후 이미지 경로
cell_path = 'cell_img/'  # 셀구역을 표현한 이미지
split_path = 'sp_img/'   # 셀단위로 쪼개진 이미지
filename = 'image02.png' # 파일명

def imshow(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
def main():
    
    img = cv2.imread(path+filename, cv2.IMREAD_GRAYSCALE)  # 이미지 불러오기
    bi_thr = 200
    ret, bi_img = cv2.threshold(img, bi_thr, 255, cv2.THRESH_BINARY)  # 이진화
    res = get_box(img, white_back=False
                  , margin=0, min_height=20
                  , line_color=(255, 0, 0)
                  , line_width=2)
    
    kernel_erode = np.ones((3, 3), np.uint8)
    erotion = cv2.erode(bi_img, kernel_erode, iterations=3)

    kernel_del_row = np.ones((1, 30), np.uint8)
    kernel_del_col = np.ones((30, 1), np.uint8)

    row_img = cv2.dilate(erotion, kernel_del_row, iterations=5)
    col_img = cv2.dilate(erotion, kernel_del_col, iterations=6)
    
    rm_line_img = copy.deepcopy(bi_img)
    
    for row_id in range(bi_img.shape[0]):
        for col_id in range(bi_img.shape[1]):
            if row_img[row_id, col_id] == 0:
                rm_line_img[row_id, col_id - 100: col_id + 100] = 255
            if col_img[row_id, col_id] == 0:
                rm_line_img[row_id - 100: row_id + 100, col_id] = 255
                
    erod_text = np.ones((2, 3) , np.uint8)
    dilate_text = np.ones((2, 3) , np.uint8)

    er_text_img = cv2.erode(rm_line_img, erod_text, iterations=5)
    er_text_img = cv2.dilate(er_text_img, dilate_text, iterations=1)
    # er_text_img = cv2.erode(er_text_img, erod_text, iterations=2)
    # er_text_img = cv2.dilate(er_text_img, dilate_text, iterations=2)

    er_text_img = draw_edge(er_text_img, margin=1)  # 이미지 가장자리 흰색으로 칠함
    
    box = get_box(er_text_img, white_back=False, margin=0, min_height=20, line_color=(255, 0, 0), line_width=2)
    box_img = box['img']
    box_coord = box['coord']

    cell = get_cell(bi_img, row_img, col_img, min_size=30)
    cell_img = cell['img']
    line_coord_x_ls = cell['line_coord_x']
    line_coord_y_ls = cell['line_coord_y']
    cell_id_ls = cell['cell_id']
    cell_coord_ls = cell['cell_coord']
    
    imshow(cell_img)
    
if __name__ == '__main__':
    main()