import os.path
import pathlib
import shutil
pic = pathlib.Path(r"E:\dm_all\dm_dataset_with_dm_ng\ng").rglob("*.bmp")

for pic_ in pic:
    name = os.path.basename(pic_)
    out_name = name.split('_')[0]
    shutil.move(pic_,f"F:\qr\dm_ok/{out_name}.jpg")