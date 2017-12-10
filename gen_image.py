import torch.nn.parallel
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import cPickle as pickle
import module
from PIL import Image
from torch.utils.data.dataset import Dataset

waterNet = module.SRResNet()
waterNet = waterNet.cuda()


model_name = 'out/model/waterNet_epoch_240.pth'
print model_name
waterNet.load_state_dict(torch.load(model_name))



image_path = '/home/jinlin/water_mark/data/'
f = open('/home/jinlin/water_mark/data/test_list.pkl', 'r')
image_list = pickle.load(f)
f.close()

class TestDataset(Dataset):
    def __init__(self, image_path, image_list, transform=None):
        self.img_path = image_path
        self.image_list = image_list
        self.transform = transform

    def __getitem__(self, index):
        mark_name = self.img_path + 'mask_face/' + self.image_list[index]
        img = Image.open(mark_name)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        clean_face_name = self.img_path + 'clean_face/' + self.image_list[index][:-6] + '.jpg'
        clean_face = Image.open(clean_face_name)
        clean_face = clean_face.convert('RGB')
        if self.transform is not None:
            clean_face = self.transform(clean_face)


        return img, clean_face, mark_name, clean_face_name

    def __len__(self):
        return len(self.image_list)


transformations = transforms.Compose([transforms.Scale(120), transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])

dset_train = TestDataset(image_path=image_path, image_list=image_list, transform=transformations)



test_loader = DataLoader(dset_train, batch_size=1, shuffle=False, num_workers=10)


for idx, (mark_face, clean_face, mark_name, clean_name) in enumerate(test_loader):

    print idx,'/',len(test_loader)

    mark_img = mark_face.mul(0.5).add(0.5)
    vutils.save_image(mark_img, 'out/gen2/'+str(idx)+'_0'+'.jpg')
    clean_img = clean_face.mul(0.5).add(0.5)
    vutils.save_image(clean_img, 'out/gen2/'+str(idx) +'_2'+ '.jpg')

    test_v = Variable(mark_face, volatile=True).cuda()
    test_result = waterNet(test_v)
    vutils.save_image(test_result.cpu().data.mul(0.5).add(0.5), 'out/gen2/'+str(idx)+'_1'+'.jpg')





















