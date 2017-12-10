import numpy as np
from PIL import Image
import cPickle as pickle
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class WaterMarkDataset(Dataset):
    def __init__(self, face_path, feature_path, mark_face_list, feature_dict, transform=None):
        self.face_path = face_path
        self.feature_path = feature_path
        self.mark_face_list = mark_face_list
        self.feature_dict = feature_dict
        self.transform = transform
        #self.transform2 = transform[1]

    def __getitem__(self, index):
        mark_face_name = self.mark_face_list[index]
        mark_face_name = self.face_path + 'mask_face/' + mark_face_name
        mark_face = Image.open(mark_face_name)
        mark_face = mark_face.convert('RGB')

        clean_face_name = self.mark_face_list[index][:6] + '.jpg'
        clean_face_name = self.face_path + 'clean_face/' + clean_face_name
        clean_face = Image.open(clean_face_name)
        clean_face = clean_face.convert('RGB')

        mask_name = self.mark_face_list[index]
        mask_name = self.face_path + 'mask_match/' + mask_name
        mask = Image.open(mask_name)
        mask = mask.convert('RGB')

        feature_name = self.face_path + 'clean_face/' + self.mark_face_list[index][:6] + '.jpg'
        feature = self.feature_dict[feature_name]


        if self.transform is not None:
            mark_face = self.transform(mark_face)
            clean_face = self.transform(clean_face)
            mask = self.transform(mask)

        return mark_face, clean_face, mask, feature

    def __len__(self):
        return len(self.mark_face_list)



def pickleload(pickle_path):
    f = open(pickle_path, 'r')
    im_list = pickle.load(f)
    return im_list

def feature_find(feature_dict, feature_name):
    featues = []
    for item in feature_name:
        array = feature_dict[item]
        feature_new = np.array(array).reshape((1, 512))
        featues.append(feature_new)
    return featues


face_path = '/home/jinlin/water_mark/data/'
feature_path = '/home/jinlin/water_mark/caffe_model/temp/'
# changge the pathes to your own pathes


mark_face = pickleload(face_path+'train_list.pkl')
feature_dict = pickleload(feature_path + 'clean_feature.pkl')
print len(mark_face)


transformations = transforms.Compose([transforms.Scale(120), transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])#, transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)


dset_train = WaterMarkDataset(face_path=face_path, feature_path=feature_path,
                              mark_face_list=mark_face, feature_dict=feature_dict, transform=transformations)


train_loader = DataLoader(dset_train, batch_size=64, shuffle=False, num_workers=16)










