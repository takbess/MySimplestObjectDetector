from pycocotools.coco import COCO
import os,time
import matplotlib.pyplot as plt
import random
from PIL import Image

def show_anns(json_path,img_dir,NUM=5):
    coco = COCO(json_path)
    image_ids = coco.getImgIds()
    # get NUM number images from image_ids
    image_ids = random.choices(image_ids,k=NUM)
    image_infos = coco.loadImgs(image_ids)

    # plot
    for image_info in image_infos:
        img_file = image_info['file_name']
        img_path = os.path.join(img_dir,img_file)
        img = Image.open(img_path)
        plt.imshow(img)
        annIds = coco.getAnnIds(imgIds=image_info['id'])
        anns = coco.loadAnns(annIds)

        for ann in anns:
            x,y,w,h = ann['bbox']
            plt.plot([x,x+w,x+w,x,x],[y,y,y+h,y+h,y],color='white')
            catName = coco.loadCats(ann['category_id'])[0]['name']
            plt.text(x+5,y-10,catName,fontsize=15,color="white")

        plt.savefig("aaa.jpg")
        plt.show()
        plt.clf()
        time.sleep(0.5)

# cocoapi version. I do not like this...
def show_anns_cocoapi(json_path,img_dir,NUM=5):
    from pycocotools.coco import COCO
    coco = COCO(json_path)
    image_ids = coco.getImgIds()
    # get NUM number images from image_ids
    import random
    image_ids = random.choices(image_ids,k=NUM)
    image_infos = coco.loadImgs(image_ids)

    # plot
    import os,time
    from PIL import Image
    import matplotlib.pyplot as plt
    for image_info in image_infos:
        img_file = image_info['file_name']
        img_path = os.path.join(img_dir,img_file)
        img = Image.open(img_path)
        plt.imshow(img)
        annIds = coco.getAnnIds(imgIds=image_info['id'])
        anns = coco.loadAnns(annIds)

        coco.showAnns(anns,draw_bbox=True,)

        plt.savefig("aaa.jpg")
        plt.clf()
        time.sleep(0.5)