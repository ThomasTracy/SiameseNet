
import os
from PIL import Image

from_dir = "/home/tracy/data/TrafficSign_single/Images_ppm"
to_dir = "/home/tracy/data/TrafficSign_single/Images"

def ppm2jpg(dir1, dir2):
    for dir in os.listdir(dir1):
        ppm_dir = os.path.join(dir1, dir)
        jpg_dir = os.path.join(dir2, dir)
        if not os.path.exists(jpg_dir):
            os.makedirs(jpg_dir)
        if os.path.isdir(ppm_dir):
            for ppm in os.listdir(ppm_dir):
                jpg_name = ppm.split('.')[0]
                if ppm.endswith('.ppm'):
                    im = Image.open(os.path.join(ppm_dir, ppm))
                    if im.size[0] >= 64:
                        jpg_file = os.path.join(jpg_dir, jpg_name + '.jpg')
                        im.save(jpg_file)
            print(dir)



if __name__ == '__main__':
    # ppm2jpg(from_dir, to_dir)
    img_dir = '/home/tracy/data/TrafficSign_single/Images/00000/00000_00024.jpg'
    im = Image.open(img_dir)
    print(im)