import cv2

image = cv2.imread("/home/user/test.jpg")

cv2.imshow('Preview' ,image)
cv2.waitKey(0)
cv2.destroyAllWindows()