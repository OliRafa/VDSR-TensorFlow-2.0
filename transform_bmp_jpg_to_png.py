import pathlib
from PIL import Image


data_dir = pathlib.Path('/mnt/hdd_raid/datasets/VDSR_Train_Dataset/Set14')
new_dir = pathlib.Path('/mnt/hdd_raid/datasets/VDSR_Train_Dataset/PNG/Set14')
#if not new_dir:
#    pathlib.Path(
#        '/mnt/hdd_raid/datasets/VDSR_Train_Dataset/PNG/' + dir_name
#    ).mkdir(
#        parents=True,
#        exist_ok=True
#    )

for image in list(data_dir.glob('*.bmp')):
    img = Image.open(image)
    output, _ = image.parts[-1].split('.')
    img.save(str(new_dir) + '/' + output + '.png')
