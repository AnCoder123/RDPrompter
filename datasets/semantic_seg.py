import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation, VisionDataset
import numpy as np
import random
import cv2
from torchvision import transforms
import torch
from datasets.util import traintransform
import torch.nn.functional as F
import imgaug.augmenters as iaa
from imgaug import SegmentationMapsOnImage  
import cv2

class BaseSemanticDataset(VisionDataset):
    """
    if you want to customize a new dataset to train the segmentation task,
    the img and mask file need be arranged as this sturcture.
        ├── data
        │   ├── my_dataset
        │   │   ├── img
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann
        │   │   │   ├── train
        │   │   │   │   ├── xxx{ann_suffix}
        │   │   │   │   ├── yyy{ann_suffix}
        │   │   │   │   ├── zzz{ann_suffix}
        │   │   │   ├── val
    """

    def __init__(self, metainfo, dataset_dir, transform, target_transform,
                 image_set='train',
                 img_suffix='.jpg',
                 ann_suffix='.png',
                 data_prefix: dict = dict(img_path='img', ann_path='ann'),
                 return_dict=False):
        '''

        :param metainfo: meta data in original dataset, e.g. class_names
        :param dataset_dir: the path of your dataset, e.g. data/my_dataset/ by the stucture tree above
        :param image_set: 'train' or 'val'
        :param img_suffix: your image suffix
        :param ann_suffix: your annotation suffix
        :param data_prefix: data folder name, as the tree shows above, the data_prefix of my_dataset: img_path='img' , ann_path='ann'
        :param return_dict: return dict() or tuple(img, ann)
        '''
        super(BaseSemanticDataset, self).__init__(root=dataset_dir, transform=transform,
                                                  target_transform=target_transform)

        self.class_names = metainfo['class_names']
        self.img_path = os.path.join(dataset_dir, data_prefix['img_path'], image_set)
        self.ann_path = os.path.join(dataset_dir, data_prefix['ann_path'], image_set)
        print('img_folder_name: {img_folder_name}, ann_folder_name: {ann_folder_name}'.format(
            img_folder_name=self.img_path, ann_folder_name=self.ann_path))
        self.img_names = [img_name.split(img_suffix)[0] for img_name in os.listdir(self.img_path) if
                          img_name.endswith(img_suffix)]
        self.img_suffix = img_suffix
        self.ann_suffix = ann_suffix
        self.return_dict = return_dict

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_names[index] + self.img_suffix))
        ann = Image.open(os.path.join(self.ann_path, self.img_names[index] + self.ann_suffix))
        if self.transforms is not None:
            img, ann = self.transforms(img, ann)
        ann = np.array(ann)

        if self.return_dict:
            data = dict(img_name=self.img_names[index], img=img, ann=ann,
                        img_path=os.path.join(self.img_path, self.img_names[index] + self.img_suffix),
                        ann_path=os.path.join(self.ann_path, self.img_names[index] + self.ann_suffix))
            return data
        return img, ann

    def __len__(self):
        return len(self.img_names)


class VOCSemanticDataset(Dataset):
    def __init__(self, root_dir, domain, transform, with_id=False, with_mask=False):
        super(VOCSemanticDataset, self).__init__()
        self.root_dir = root_dir

        self.image_dir = self.root_dir + 'JPEGImages/'
        self.xml_dir = self.root_dir + 'Annotations/'
        self.mask_dir = self.root_dir + 'SegmentationClass/'

        self.image_id_list = [image_id.strip() for image_id in open('./data/%s.txt' % domain).readlines()]
        self.transform = transform
        self.with_id = with_id
        self.with_mask = with_mask
        self.class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person',
                            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __len__(self):
        return len(self.image_id_list)

    def get_image(self, image_id):
        image = Image.open(self.image_dir + image_id + '.jpg').convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def get_mask(self, image_id):
        mask_path = self.mask_dir + image_id + '.png'
        if os.path.isfile(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = None
        return mask

    def __getitem__(self, index):
        image_id = self.image_id_list[index]

        data_list = [self.get_image(image_id)]

        if self.with_id:
            data_list.append(image_id)

        if self.with_mask:
            data_list.append(self.get_mask(image_id))

        return data_list


class TorchVOCSegmentation(VOCSegmentation):
    def __init__(self, root, year='2012', image_set='train', download=False, transform=None, target_transform=None):
        super(TorchVOCSegmentation, self).__init__(root=root, year=year, image_set=image_set, download=download,
                                                   transform=transform, target_transform=target_transform)
        self.class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person',
                            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        target = np.array(target)
        return img, target




class DemoSegmentation(VisionDataset):
    """
        ├── data
        │   ├── my_dataset
        │   │   ├── img
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann
        │   │   │   ├── train
        │   │   │   │   ├── xxx{ann_suffix}
        │   │   │   │   ├── yyy{ann_suffix}
        │   │   │   │   ├── zzz{ann_suffix}
        │   │   │   ├── val
    """

    def __init__(self, metainfo, dataset_dir, image_set='train', transform=None, target_transform=None,
                 img_suffix='.jpg',
                 ann_suffix='.png',
                 data_prefix: dict = dict(img_path='img', ann_path='ann'),
                 return_dict=False):
        '''

        :param metainfo: meta data in original dataset, e.g. class_names
        :param dataset_dir: the path of your dataset, e.g. data/my_dataset/ by the stucture tree above
        :param image_set: 'train' or 'val'
        :param img_suffix: your image suffix
        :param ann_suffix: your annotation suffix
        :param data_prefix: data folder name, as the tree shows above, the data_prefix of my_dataset: img_path='img' , ann_path='ann'
        :param return_dict: return dict() or tuple(img, ann)
        '''
        super(DemoSegmentation, self).__init__(root=dataset_dir, transform=transform,
                                                  target_transform=target_transform)

        self.class_names = [metainfo['class_names']]
        

        self.img_path = os.path.join(dataset_dir, data_prefix['img_path'], image_set)
        self.ann_path = os.path.join(dataset_dir, data_prefix['ann_path'], image_set)

        print('img_folder_name: {img_folder_name}, ann_folder_name: {ann_folder_name}'.format(
            img_folder_name=self.img_path, ann_folder_name=self.ann_path))
        self.img_names = [img_name.split(img_suffix)[0] for img_name in os.listdir(self.img_path) if
                          img_name.endswith(img_suffix)]
        
        self.img_suffix = img_suffix
        self.ann_suffix = ann_suffix
        self.return_dict = return_dict

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_names[index] + self.img_suffix))
        ann = Image.open(os.path.join(self.ann_path, self.img_names[index] + self.ann_suffix))
        if self.transforms is not None:
            img, ann = self.transforms(img, ann)
        ann = np.array(ann)

        # if self.return_dict:
        #     data = dict(img_name=self.img_names[index], img=img, ann=ann,
        #                 img_path=os.path.join(self.img_path, self.img_names[index] + self.img_suffix),
        #                 ann_path=os.path.join(self.ann_path, self.img_names[index] + self.ann_suffix))
        #     return data
        
        return img, ann

    def __len__(self):
        return len(self.img_names)


# MVTec-Unseen  
class FewShotSegmentation(VisionDataset):

    def __init__(self, metainfo, dataset_dir,state='train',n_ways=1, n_shots=5, transform=None, target_transform=None,
                 img_suffix='.png',
                 ann_suffix='.png',
                 data_prefix: dict = dict(img_path='test', ann_path='ground_truth'),
                 return_dict=False):
        
        super(FewShotSegmentation, self).__init__(root=dataset_dir, transform=transform,
                                                  target_transform=target_transform)

        self.class_names = [metainfo['class_names']]  
        subClassNames = [f for f in os.listdir(dataset_dir)]
        self.finalIndex=[]
        self.img_names=[]  
        self.gt_names=[] 
        self.state=state
        self.n_ways=n_ways
        self.n_shots=n_shots

        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]

        train_transformlist= [
            traintransform.RandRotate([-10, 10], padding=mean, ignore_label=255),
            traintransform.RandomGaussianBlur(),
            traintransform.RandomHorizontalFlip(),
            traintransform.RandomVerticalFlip(),
            traintransform.ToTensor(),
            traintransform.Normalize(mean=mean, std=std)]
        self.train_transform = traintransform.Compose(train_transformlist)

        val_transformlist = [
            traintransform.ToTensor(),
            traintransform.Normalize(mean=mean, std=std)]
        self.val_transform = traintransform.Compose(val_transformlist)


        self.augmenter = iaa.Sequential([
            iaa.CoarseDropout(0.1, size_percent=0.2),
            iaa.GaussianBlur((0, 3.0)),
            iaa.Affine(translate_px={"x": (-40, 40)}),
            iaa.Crop(px=(0, 10)),
            iaa.ElasticTransformation(alpha=10, sigma=1),
            iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
        ])

        if state=='train':
            for subClassName  in subClassNames:
                
                if subClassName==self.class_names[0]:
                    continue

                self.img_path = os.path.join(dataset_dir,subClassName, data_prefix['img_path'])
                self.ann_path = os.path.join(dataset_dir,subClassName, data_prefix['ann_path'])
                subFileNames = [f for f in os.listdir(self.img_path)]  
                for subFilename in subFileNames:
                    if subFilename=='good':
                        continue
                    self.img_single_names = [os.path.join(self.img_path,subFilename,img_name) for img_name in os.listdir(os.path.join(self.img_path,subFilename)) 
                                      if img_name.endswith(img_suffix)]
                    self.gt_single_names = [os.path.join(self.ann_path,subFilename,img_name[:-4]+'_mask.png') for img_name in os.listdir(os.path.join(self.img_path,subFilename)) 
                                      if img_name.endswith(img_suffix)]
                    self.img_names.append(self.img_single_names)
                    self.gt_names.append(self.gt_single_names)

        if state=='test':
            self.img_path = os.path.join(dataset_dir,self.class_names[0], data_prefix['img_path'])
            self.ann_path = os.path.join(dataset_dir,self.class_names[0], data_prefix['ann_path'])
            subFileNames = [f for f in os.listdir(self.img_path)]  
            
            subFileNames.remove('good')
            
            for subFilename in subFileNames:
                classNum=subFileNames.index(subFilename)+1
                if subFilename=='good':
                    continue
                self.img_single_names =[os.path.join(self.img_path,subFilename,img_name) for img_name in os.listdir(os.path.join(self.img_path,subFilename)) 
                                    if img_name.endswith(img_suffix)]
                self.gt_single_names =[os.path.join(self.ann_path,subFilename,img_name[:-4]+'_mask.png') for img_name in os.listdir(os.path.join(self.img_path,subFilename)) 
                                      if img_name.endswith(img_suffix)]
                
                # random.seed(123)
                # supportImgPaths=random.sample(self.img_single_names,self.n_shots)
                supportImgPaths=self.img_single_names[:self.n_shots]
                supportImgPaths_indices = [self.img_single_names.index(element) for element in supportImgPaths]
                supportMaskPaths = [self.gt_single_names[index] for index in supportImgPaths_indices]

                supportImgs=[]
                supportMasks=[]
                for supportImgPath in supportImgPaths:
                    imgname=supportImgPath.split('/')[-1][-7:-4]
                    maskname=supportMaskPaths[supportImgPaths.index(supportImgPath)].split('/')[-1][-12:-9]
                    assert imgname==maskname,"x must be equal to y" 
                    img = Image.open(supportImgPath).convert('RGB').resize((1024,1024))
                    mask= Image.open(supportMaskPaths[supportImgPaths.index(supportImgPath)]).resize((1024,1024))
                    image = np.asarray(img, dtype=np.uint8)
                    segmap = np.asarray(mask, dtype=np.int32)

                    segmap = SegmentationMapsOnImage(segmap, shape=image.shape)
                    images_aug, segmaps_aug = self.augmenter(image=image, segmentation_maps=segmap)
                    img = Image.fromarray(images_aug).convert('RGB').resize((1024,1024))
                    mask= Image.fromarray(segmaps_aug.get_arr()).convert('L').resize((256,256))
                    
                    if self.transforms is not None:
                        img, mask = self.val_transform(np.array(img), np.array(mask))
                        img = np.array(img)
                        mask = np.array(mask)
                        mask[np.where(mask>0)]=1   

                        supportImgs.append(img)
                        supportMasks.append(mask)

                otherImgPaths = [x for x in self.img_single_names if x not in supportImgPaths]
                otherMaskPaths = [x for x in self.gt_single_names if x not in supportMaskPaths]

                for queryImgpaths in otherImgPaths:
                    queryImgs=[]
                    queryMasks=[]
                    queryImgpaths_indices = [otherImgPaths.index(element) for element in [queryImgpaths]]
                    queryLabelpaths = [otherMaskPaths[index] for index in queryImgpaths_indices]
                    imgname=[queryImgpaths][0].split('/')[-1][-7:-4]
                    maskname=queryLabelpaths[0].split('/')[-1][-12:-9]
                    assert imgname==maskname,"x must be equal to y" 
                    img = Image.open([queryImgpaths][0]).convert('RGB').resize((1024,1024))
                    Label= Image.open(queryLabelpaths[0]).convert('L').resize((256,256))

                    queryImg, queryLabel = self.val_transform(np.array(img), np.array(Label))

                    queryImg = np.array(queryImg)
                    queryLabel = np.array(queryLabel)
                    queryLabel[np.where(queryLabel>0)]=1 

                    queryImgs.append(queryImg)
                    queryMasks.append(queryLabel)
                    self.finalIndex.append((supportImgs,supportMasks,queryImgs,queryMasks,classNum))
                    
                self.img_names.append(self.img_single_names)
                self.gt_names.append(self.gt_single_names)

        
        self.img_suffix = img_suffix
        self.ann_suffix = ann_suffix
        self.return_dict = return_dict

    def __getitem__(self, index):

        if self.state=='train':
            classIndex=self.img_names.index(random.choice(self.img_names))
            classFileName=self.img_names[classIndex]  
            classMaskFileName=self.gt_names[classIndex]

            supportImgPaths=random.sample(classFileName,self.n_shots)
            supportImgPaths_indices = [classFileName.index(element) for element in supportImgPaths]
            supportMaskPaths = [classMaskFileName[index] for index in supportImgPaths_indices]
            supportImgs=[]
            supportMasks=[]
            for supportImgPath in supportImgPaths:
                imgname=supportImgPath.split('/')[-1][-7:-4]
                maskname=supportMaskPaths[supportImgPaths.index(supportImgPath)].split('/')[-1][-12:-9]
                assert imgname==maskname,"x must be equal to y" 
                img = Image.open(supportImgPath).convert('RGB').resize((1024,1024))
                mask= Image.open(supportMaskPaths[supportImgPaths.index(supportImgPath)]).convert('L').resize((1024,1024))
                if self.transforms is not None:
                    img, mask = self.train_transform(np.array(img), np.array(mask))
                    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), (256,256), mode='nearest').byte().squeeze()
                    img = np.array(img)
                    mask = np.array(mask)
                    mask[np.where(mask>0)]=1
                    supportImgs.append(img)
                    supportMasks.append(mask)

            queryImgs=[]
            queryMasks=[]
            otherImgPaths = [x for x in classFileName if x not in supportImgPaths]
            otherMaskPaths = [x for x in classMaskFileName if x not in supportMaskPaths]
            queryImgpaths = random.sample(otherImgPaths, 1)
            queryImgpaths_indices = [otherImgPaths.index(element) for element in queryImgpaths]
            queryLabelpaths = [otherMaskPaths[index] for index in queryImgpaths_indices]
            imgname=queryImgpaths[0].split('/')[-1][-7:-4]
            maskname=queryLabelpaths[0].split('/')[-1][-12:-9]
            assert imgname==maskname,"x must be equal to y" 

            img = Image.open(queryImgpaths[0]).convert('RGB').resize((1024,1024))
            Label= Image.open(queryLabelpaths[0]).convert('L').resize((1024,1024))
            queryImg, queryLabel = self.train_transform(np.array(img), np.array(Label))
            queryLabel = F.interpolate(queryLabel.unsqueeze(0).unsqueeze(0).float(), (256,256), mode='nearest').byte().squeeze()

            queryImg = np.array(queryImg)
            queryLabel = np.array(queryLabel)
            queryLabel[np.where(queryLabel>0)]=1 
            queryImgs.append(queryImg)
            queryMasks.append(queryLabel)
            classNum=0

        if self.state=="test":
            (supportImgs,supportMasks,queryImgs,queryMasks,classNum)=self.finalIndex[index]
        
        return (supportImgs,supportMasks,queryImgs,queryMasks,classNum)


    

    def __len__(self):
        if self.state=='train':
            return len(self.img_names)
        else:
            return len(self.finalIndex)
    


# SDD  FSSD-12 CID
class FewShotSegmentation2(VisionDataset):

    def __init__(self, metainfo, dataset_dir,state='train',n_ways=1, n_shots=5, transform=None, target_transform=None,
                 img_suffix='.jpg',
                 ann_suffix='.png',
                 data_prefix: dict = dict(img_path='test', ann_path='ground_truth'),
                 return_dict=False):
        
        super(FewShotSegmentation2, self).__init__(root=dataset_dir, transform=transform,
                                                  target_transform=target_transform)

        self.class_names = [metainfo['class_names']]  
        subClassNames = [f for f in os.listdir(dataset_dir)]

        self.finalIndex=[]
        self.img_names=[]  
        self.gt_names=[] 
        self.state=state
        self.n_ways=n_ways
        self.n_shots=n_shots

        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]

        train_transformlist= [
            # traintransform.RandScale([0.9, 1.1]),
            traintransform.RandRotate([-10, 10], padding=mean, ignore_label=255),
            traintransform.RandomGaussianBlur(),
            traintransform.RandomHorizontalFlip(),
            traintransform.RandomVerticalFlip(),
            # traintransform.Crop([1024, 1024], crop_type='rand', padding=mean, ignore_label=255),
            traintransform.ToTensor(),
            traintransform.Normalize(mean=mean, std=std)]
        self.train_transform = traintransform.Compose(train_transformlist)

        val_transformlist = [
            traintransform.ToTensor(),
            traintransform.Normalize(mean=mean, std=std)]
        self.val_transform = traintransform.Compose(val_transformlist)



        if state=='train':
            for subClassName  in subClassNames:
                if subClassName==self.class_names[0]:
                    continue

                self.img_path = os.path.join(dataset_dir,subClassName, data_prefix['img_path'])
                self.ann_path = os.path.join(dataset_dir,subClassName, data_prefix['ann_path'])
                subFileNames = [f for f in os.listdir(self.img_path)]  
                for subFilename in subFileNames:
                    if subFilename=='good':
                        continue

                    self.img_single_names = [os.path.join(self.img_path,subFilename,img_name) for img_name in os.listdir(os.path.join(self.img_path,subFilename)) 
                                      if img_name.endswith(".png")]
                    self.gt_single_names = [os.path.join(self.ann_path,subFilename,img_name[:-4]+'_mask.png') for img_name in os.listdir(os.path.join(self.img_path,subFilename)) 
                                      if img_name.endswith(".png")]
                    img_suffix=".png"

                    self.img_names.append(self.img_single_names)
                    self.gt_names.append(self.gt_single_names)

        if state=='test':
            self.img_path = os.path.join(dataset_dir,self.class_names[0], data_prefix['img_path'])
            self.ann_path = os.path.join(dataset_dir,self.class_names[0], data_prefix['ann_path'])
            subFileNames = [f for f in os.listdir(self.img_path)]  

            
            for subFilename in subFileNames:
                # print(subFilename)
                classNum=subFileNames.index(subFilename)+1
                if subFilename=='good':
                    continue

                self.img_single_names =[os.path.join(self.img_path,subFilename,img_name) for img_name in os.listdir(os.path.join(self.img_path,subFilename)) 
                                    if img_name.endswith(img_suffix)]
                self.gt_single_names =[os.path.join(self.ann_path,subFilename,img_name[:-4]+'.png') for img_name in os.listdir(os.path.join(self.img_path,subFilename)) 
                                      if img_name.endswith(img_suffix)]
                
                # random.seed(123)
                # supportImgPaths=random.sample(self.img_single_names,self.n_shots)
                supportImgPaths=self.img_single_names[:self.n_shots]

                supportImgPaths_indices = [self.img_single_names.index(element) for element in supportImgPaths]
                supportMaskPaths = [self.gt_single_names[index] for index in supportImgPaths_indices]
                supportImgs=[]
                supportMasks=[]
                for supportImgPath in supportImgPaths:
                    imgname=supportImgPath.split('/')[-1][:-4]
                    maskname=supportMaskPaths[supportImgPaths.index(supportImgPath)].split('/')[-1][:-4]
                    # assert imgname==maskname,"x must be equal to y" 
                    img = Image.open(supportImgPath).convert('RGB').resize((1024, 1024))
                    mask= Image.open(supportMaskPaths[supportImgPaths.index(supportImgPath)]).convert("L").resize((256, 256))
                    if self.transforms is not None:
                        img, mask = self.val_transform(np.array(img), np.array(mask))
                        img = np.array(img)
                        mask = np.array(mask)
                        mask[np.where(mask>0)]=1   
                        supportImgs.append(img)
                        supportMasks.append(mask)

                otherImgPaths = [x for x in self.img_single_names if x not in supportImgPaths]
                otherMaskPaths = [x for x in self.gt_single_names if x not in supportMaskPaths]
                for queryImgpaths in otherImgPaths:
                    queryImgs=[]
                    queryMasks=[]
                    queryImgpaths_indices = [otherImgPaths.index(element) for element in [queryImgpaths]]
                    queryLabelpaths = [otherMaskPaths[index] for index in queryImgpaths_indices]
                    imgname=[queryImgpaths][0].split('/')[-1][:-4]
                    maskname=queryLabelpaths[0].split('/')[-1][:-4]
                    img = Image.open([queryImgpaths][0]).convert('RGB').resize((1024, 1024))
                    Label= Image.open(queryLabelpaths[0]).convert("L").resize((256, 256))

                    queryImg, queryLabel = self.val_transform(np.array(img), np.array(Label))
                    queryImg = np.array(queryImg)
                    queryLabel = np.array(queryLabel)
                    queryLabel[np.where(queryLabel>0)]=1   
                    queryImgs.append(queryImg)
                    queryMasks.append(queryLabel)

                    self.finalIndex.append((supportImgs,supportMasks,queryImgs,queryMasks,classNum,supportImgPaths,[queryImgpaths][0]))
                    
                self.img_names.append(self.img_single_names)
                self.gt_names.append(self.gt_single_names)

        
        self.img_suffix = img_suffix
        self.ann_suffix = ann_suffix
        self.return_dict = return_dict

    def __getitem__(self, index):

        if self.state=='train':
            classIndex=self.img_names.index(random.choice(self.img_names))
            classFileName=self.img_names[classIndex]   
            classMaskFileName=self.gt_names[classIndex] 

            # choose support image and support Mask
            supportImgPaths=random.sample(classFileName,self.n_shots)
            supportImgPaths_indices = [classFileName.index(element) for element in supportImgPaths]
            supportMaskPaths = [classMaskFileName[index] for index in supportImgPaths_indices]
            supportImgs=[]
            supportMasks=[]
            for supportImgPath in supportImgPaths:
                imgname=supportImgPath.split('/')[-1][:-4]
                maskname=supportMaskPaths[supportImgPaths.index(supportImgPath)].split('/')[-1][:-4]

                img = Image.open(supportImgPath).convert('RGB').resize((1024, 1024))
                mask= Image.open(supportMaskPaths[supportImgPaths.index(supportImgPath)]).convert("L").resize((1024, 1024))
                if self.transforms is not None:
                    img, mask = self.train_transform(np.array(img), np.array(mask))
                    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), (256,256), mode='nearest').byte().squeeze()
                    
                    img = np.array(img)
                    mask = np.array(mask)
                    mask[np.where(mask>0)]=1  
                    supportImgs.append(img)
                    supportMasks.append(mask)

            queryImgs=[]
            queryMasks=[]
            otherImgPaths = [x for x in classFileName if x not in supportImgPaths]
            otherMaskPaths = [x for x in classMaskFileName if x not in supportMaskPaths]
            queryImgpaths = random.sample(otherImgPaths, 1)
            queryImgpaths_indices = [otherImgPaths.index(element) for element in queryImgpaths]
            queryLabelpaths = [otherMaskPaths[index] for index in queryImgpaths_indices]
            imgname=queryImgpaths[0].split('/')[-1][:-4]
            maskname=queryLabelpaths[0].split('/')[-1][:-4]
            
            img = Image.open(queryImgpaths[0]).convert('RGB').resize((1024, 1024))
            Label= Image.open(queryLabelpaths[0]).convert("L").resize((1024, 1024))
            queryImg, queryLabel = self.train_transform(np.array(img), np.array(Label))
            queryLabel = F.interpolate(queryLabel.unsqueeze(0).unsqueeze(0).float(), (256,256), mode='nearest').byte().squeeze()

            queryImg = np.array(queryImg)
            queryLabel = np.array(queryLabel)
            queryLabel[np.where(queryLabel>0)]=1  

            queryImgs.append(queryImg)
            queryMasks.append(queryLabel)
            classNum=0

        if self.state=="test":
            (supportImgs,supportMasks,queryImgs,queryMasks,classNum,supportImgPaths,queryImgpaths)=self.finalIndex[index]
           
        return (supportImgs,supportMasks,queryImgs,queryMasks, classNum)

    def __len__(self):
        if self.state=='train':
            return len(self.img_names)
        else:
            return len(self.finalIndex)
