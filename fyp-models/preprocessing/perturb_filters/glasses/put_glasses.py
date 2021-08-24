import cv2

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def put_hat(hat, fc, x, y, w, h):
    face_width = w
    face_height = h
    hat_width = face_width + 1
    hat_height = int(0.50 * face_height) + 1
    hat = cv2.resize(hat, (hat_width, hat_height))

    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if hat[i][j][k] < 235:
                    fc[y + i - int(0.40 * face_height)][x + j][k] = hat[i][j][k]
    return fc


def apply_glasses(ori_path, image_fn, target_glasses):
    """
    
    ori_path : str
        Original folder path
    image_fn : str
        File path of target image
    target_glasses: str
        Target name for new image file
    """
    # Call apply_glasses(ori_path, image_fn, target_glasses)
    filename=ori_path + image_fn
    img=cv2.imread(filename)
    
    glass=cv2.imread('glasses.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face_width = w
        face_height = h
        hat_width = face_width + 1
        hat_height = int(0.50 * face_height) + 1
        
        glass = cv2.resize(glass, (hat_width, hat_height))
        
        for i in range(hat_height):
            for j in range(hat_width):
                for k in range(3):
                    if glass[i][j][k] < 235:
                        img[y + i - int(-0.20 * face_height)][x + j][k] = glass[i][j][k]
        cv2.imwrite(target_glasses+image_fn, img)
