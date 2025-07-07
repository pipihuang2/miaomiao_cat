import requests
import os
import tqdm
import shutil
#通过第一个URL 收集来获取NG的id
#通过id来抓取地址



# 这里第一步先把NG的 uid查找出来，然后根据UID进去把数据拉回来
# 你抓到的接口 URL（可以替换为其他ID的接口也行）
url = "http://192.168.0.67/api/v1/inspection-results/257106/image-files"
product_id = []
for i in tqdm.tqdm(range(99)):
    page_num = i

    url2 = f"http://192.168.0.67/api/v1/inspection-results?page={page_num}&page_size=10&ok=0&product=800B"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    # 发请求获取 JSON 数据
    res = requests.get(url2, headers=headers)
    res.raise_for_status()  # 遇到错误会抛出异常

    datas = res.json()
    for data in datas["data"]['items']:
        id_ = data["id"]
        product_id.append(id_) #获取每页上的id
print('一共有',len(product_id))


for id in product_id:
    url = f"http://192.168.0.67/api/v1/inspection-results/{id}/image-files"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    # 发请求获取 JSON 数据
    res = requests.get(url, headers=headers)
    res.raise_for_status()  # 遇到错误会抛出异常
    datas = res.json()
    data1 = datas['data']['items'][0]['filename']
    data2 = datas['data']['items'][1]['filename']
    data1 = os.path.join("V:\saved_image",data1.split("E:\\saved_image\\")[-1])
    data2 = os.path.join("V:\saved_image", data2.split("E:\\saved_image\\")[-1])
    # if os.path.exists(data1):
    #     # 路径存在，继续处理，比如移动、复制等
    #     print(f"文件存在：{data1}{data2}")
    #     shutil.copy(data1, r"F:\NG-STAR\800B")
    #     shutil.copy(data2, r"F:\NG-STAR\800B")
    # else:
    #     # 路径不存在，跳过
    #     print(f"文件不存在，跳过：{data1}")


