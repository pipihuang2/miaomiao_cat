import os

def generate_01_matrix(content):
    from pylibdmtx.pylibdmtx import encode
    import numpy as np
    import matplotlib.pyplot as plt
    encoded = encode(content.encode('utf8'), size='14x14')
    _bytes = encoded.pixels
    img = np.frombuffer(_bytes, dtype=np.uint8).reshape(encoded.height, encoded.width, 3)
    matrix = (img[10:-10:5, 10:-10:5, 0] // 255)
    # plt.imshow(matrix, cmap='gray')
    # plt.show()
    matrix = matrix.reshape(-1).tolist()
    return matrix

def build_label_dict(images_dir):
    images = os.listdir(images_dir)
    result = {}
    for name in images:
        prefix = name.split("_")[0]
        if len(prefix) == 11:
            result[name] = generate_01_matrix(prefix)
    return result

if __name__ == '__main__':
    build_label_dict("E:\dm_all\dm_dataset_with_dm_ng\ok")
    # generate_01_matrix("06305039040")