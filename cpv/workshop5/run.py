import cv2

def align_image(img, template, max_feature, keep_percentage):
    img_gray= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    template_gray= cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

    orb = cv2.ORB_create(max_feature)
    keypoint1, descriptor1= orb.detectAndCompute(img)
