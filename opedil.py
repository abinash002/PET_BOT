import os
import sys
import dlib
import matplotlib.pyplot as plt
from PIL import Image

# adjust these variables as necessary

# dirname is the directory relative to the script where the files to detect a face and crop live
dirname = "val"
#  put_dirname is the name of the directory where the cropped images will be written, relative to the script
put_dirname = "cropped"
# the width and height in pixels of the saved image
crop_width = 108
# whether this is just a face crop (true) or whether we're trying to include other elements in the image. 
# Based on the shortest distance between the detected face square and the edge of the image
simple_crop = False


face_detector = dlib.get_frontal_face_detector()
file_types = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')

files = [file_i
         for file_i in os.listdir(dirname)
         if file_i.endswith(file_types)]

filenames = [os.path.join(dirname, fname)
             for fname in files]
             
# do face detection on the image

print('found %d files' % len(filenames))

filename_inc = 100

filecount = 1

for file in filenames:
    img = plt.imread(file)
    detected_faces = face_detector(img, 1)
    print("[%d of %d] %d detected faces in %s" % (filecount, len(filenames), len(detected_faces), file))
    for i, face_rect in enumerate(detected_faces):
        width = face_rect.right() - face_rect.left()
        height = face_rect.bottom() - face_rect.top()
        if width >= crop_width and height >= crop_width:
            image_to_crop = Image.open(file)
            
            if simple_crop:
                crop_area = (face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom())
            else:
                size_array = []
                size_array.append(face_rect.top())
                size_array.append(image_to_crop.height - face_rect.bottom())
                size_array.append(face_rect.left())
                size_array.append(image_to_crop.width - face_rect.right())
                size_array.sort()
                short_side = size_array[0]
                crop_area = (face_rect.left() - size_array[0] , face_rect.top() - size_array[0], face_rect.right() + size_array[0], face_rect.bottom() + size_array[0])

            cropped_image = image_to_crop.crop(crop_area)
            crop_size = (crop_width, crop_width)
            cropped_image.thumbnail(crop_size)
            cropped_image.save(put_dirname + "/" + str(filename_inc) + ".jpg", "JPEG")
            filename_inc += 1
    filecount += 1
