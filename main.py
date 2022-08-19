import pytesseract
import cv2
from pytesseract import Output
import numpy as np

def read_image(name, path = ''):
    im_read = cv2.imread(path + name)
    return im_read

# Scaling of image 300 DPI
def imageResize(img):
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)  # Inter Cubic
    return img


# BGR to GRAY
def bgrtogrey(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# Noise reduction
def noise_removal(img):
    #kernel = np.ones((1, 1), np.uint8)
    kernel = np.ones((1, 1), np.uint8)
    #img1 = cv2.dilate(img, kernel, iterations=1)
    img2 = cv2.erode(img, kernel, iterations=1)
    #cv2.imshow('original; image', img)
    #cv2.imshow('dialate', img1)
    #cv2.imshow('erode', img2)
    #cv2.waitKey(0)
    return img

# BGR to GRAY
def process_img(img_1):
    img = img_1
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    ret, new_img = cv2.threshold(image_final, 0, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
    '''
            line  8 to 12  : Remove noisy portion 
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
                                                         3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(new_img, kernel, iterations=5)  # dilate , more the iteration more the dilation

    # for cv2.x.x
    cv2.imshow('dilated image', dilated)
    cv2.waitKey(0)
    return dilated

def separate_text_areas(original_img, processed_img):
    clone_original = original_img
    cv2.imshow('processed_img image', processed_img)
    cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)  # findContours returns 3 variables for getting contours

    # for cv3.x.x comment above line and uncomment line below

    # image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    index = 0
    img_for_ocr = []
    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        #print('w = '+str(w), 'h = '+str(h))
        if w < 20 and h < 20:
            continue

        # draw rectangle around contour on original image
        #cv2.rectangle(clone_original, (x, y), (x + w, y + h), (255, 0, 255), 2)

        #you can crop image and send to OCR  , false detected will return no text :)
        cropped = original_img[y :y +  h , x : x + w]

        s = 'crop_' + str(index) + '.jpg'
        cv2.imwrite(s , cropped)
        img_for_ocr.append(cropped)
        index = index + 1

    print("index = "+ str(index))
    return img_for_ocr


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'

def tesseract_ocr(img, custom_config = ''):
    text_extract = pytesseract.image_to_data(img, output_type=Output.DICT, config=custom_config)
    for count, item in enumerate(text_extract['conf']):
        if float(item) > 30:
            if len(text_extract['text'][count].replace(" ", "")):
                (x, y, w, h) = (text_extract['left'][count], text_extract['top'][count], text_extract['width'][count],
                                text_extract['height'][count])
                print("Text:" + text_extract['text'][count].replace(" ", ""), "\n Conf: " + str(item),
                      " left: " + str(text_extract['left'][count]), " top: " + str(text_extract['top'][count]),
                      " width: " + str(text_extract['width'][count]),
                      " length:" + str(len(text_extract['text'][count].replace(" ", ""))))

if __name__ == "__main__":
    cv_img = read_image("im_us_1.jpg")
    processed_img = process_img(cv_img)
    ocr_img_list = separate_text_areas(cv_img, processed_img)
    #print(ocr_img_list)
    test = 0
    for image in ocr_img_list:
        processed_img = imageResize(image)
        processed_img = bgrtogrey(processed_img)
        #processed_img = noise_removal(processed_img)
        processed_img = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        processed_img = noise_removal(processed_img)
        #cv2.imshow('original; image', image)
        #cv2.imshow('processed image', processed_img)
        #cv2.waitKey(0)
        tesseract_ocr(processed_img)
        '''test+=1
        if test > 0:
            break
'''




