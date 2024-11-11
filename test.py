from read_XML import read_images_and_xml
import os

current_working_dir = os.getcwd()
dir = os.path.join(current_working_dir, 'Potholes', 'annotated-images', 'train')
_, annotations, _ = read_images_and_xml(dir)

print(f'anntations: {annotations[0]}')

for annotation in annotations[0]:
    xmin, ymin, width, height = annotation
    xmax = xmin + width
    ymax = ymin + height
    print(f'xmin: {xmin}, ymin: {ymin}, width: {width}, height: {height}')
    print(f'xmax: {xmax}, ymax: {ymax}')