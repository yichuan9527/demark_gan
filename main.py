from __future__ import print_function

import data_read
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import module
import pre_model


cudnn.benchmark = True

class CnnLoss(nn.Module):
    def __init__(self, rescaling_factor=12.75):
        super(CnnLoss, self).__init__()
        self.cnn = cnn()
        self.rescaling_factor = rescaling_factor
        self.softmax = nn.Softmax()

    def modified_euclidean_distance(self, x):
        (num_images, num) = x.size()
        print(x.size())
        return torch.sum(torch.pow(x, 2)).mul(1.0 / (num_images * self.rescaling_factor))

    def __call__(self, clean_face, move_mark_face):
        _, clean_feature, crelu = self.cnn(clean_face)
        _, mark_feature, mrelu = self.cnn(move_mark_face)

        clean_feature_maps = self.softmax(clean_feature.view(-1, 512))
        mark_feature_maps = self.softmax(mark_feature.view(-1, 512))
        return self.modified_euclidean_distance(clean_feature_maps - mark_feature_maps)


class cnn(nn.Module):
    def __init__(self, requires_grad=False):
        super(cnn, self).__init__()
        resnet = pre_model.resnet()
        resnet = resnet.cuda()
        check_point = torch.load('resnet.pth.tar')
        resnet.load_state_dict(check_point['state_dict'])
        self.modified_pretrained = resnet#nn.Sequential(*list(pretrained_model.features.children())[:-1])

        for (_, layer) in self.modified_pretrained._modules.items():
            layer.requires_grad = requires_grad

    def forward(self, x):
        return self.modified_pretrained(x)


gamma = 1
lrG = 0.00005
lrD = 0.00005
epoches = 1000
beta1 = 0.0001
beta2 = 0.0001


waterNet = module.SRResNet()
DNet = module._netD()
localNet = module.localD()

#print waterNet
#print DNet

face_path = '/home/jinlin/water_mark/data/'
feature_path = '/home/jinlin/water_mark/caffe_model/temp/'
mark_face = data_read.pickleload(face_path+'train_list.pkl')
feature_dict = data_read.pickleload(feature_path + 'clean_feature.pkl')

transformations = transforms.Compose([transforms.Scale(120),transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])


dset_train = data_read.WaterMarkDataset(face_path=face_path, feature_path=feature_path,
                              mark_face_list=mark_face, feature_dict=feature_dict, transform=transformations)



test_loader = DataLoader(dset_train,
                          batch_size=6,
                          shuffle=False,
                          num_workers=1 # 1 for CUDA
                         )

train_loader = DataLoader(dset_train,
                          batch_size=64,
                          shuffle=False,
                          num_workers=3 # 1 for CUDA
                         )


test_iter = iter(test_loader)
test_img, _, _, _ = test_iter.next()
#print type(test_img)
test_img = test_img.mul(0.5).add(0.5)
vutils.save_image(test_img, 'out/real_samples.png')

feature_loss = CnnLoss()
feature_loss = feature_loss.cuda()


L2loss = nn.MSELoss()
L2loss = L2loss.cuda()
waterNet = waterNet.cuda()
DNet = DNet.cuda()
localNet = localNet.cuda()


#load the trained model
waterNet.load_state_dict(torch.load('/home/jinlin/water_mark/new_model/feature_loss2/out/model/waterNet_epoch_250.pth'))
DNet.load_state_dict(torch.load('/home/jinlin/water_mark/new_model/feature_loss2/out/model/netD_epoch_250.pth'))
localNet.load_state_dict(torch.load('/home/jinlin/water_mark/new_model/feature_loss2/out/model/localNet_epoch_250.pth'))


g_optimizer = optim.RMSprop(waterNet.parameters(), lr=lrG)
d_optimizer = optim.RMSprop(DNet.parameters(), lr=lrD)


for epoch in range(250, epoches):
    idx = 0
    data_g = []
    data_d = []
    data_pix = []
    for idx,(img1, img2, mask, feature) in enumerate(train_loader):
        if idx > 2967:
            break

        DNet.zero_grad()
        for p in DNet.parameters():
            p.data.clamp_(-0.01, 0.01)

        mark_face = Variable(img1).cuda()
        clean_face = Variable(img2).cuda()

        move_mark = waterNet(mark_face)
        real_out = DNet(clean_face)
        fake_out = DNet(move_mark)

        gan_loss = real_out - fake_out
        pix_loss = L2loss(move_mark, clean_face)


        #local_loss
        #70*70 local area, you can adjust the size in your experiment
        local_face = img2[:, :, 25:95, 45:115]
        local_mark_face = move_mark.data[:, :, 25:95, 45:115]
        local_face_v = Variable(local_face).cuda()
        local_mark_face_v = Variable(local_mark_face)

        local_real_out = localNet(local_face_v)
        local_fake_out = localNet(local_mark_face_v)
        local_gan_loss = local_real_out - local_fake_out


        #maks_loss
        #*************************************
        loss = beta1*gan_loss + gamma*pix_loss + beta2*local_gan_loss #+ feature_loss
        errD = loss
        loss.backward()
        d_optimizer.step()

#****************************************************
        for p in DNet.parameters():
            p.requires_grad = False

        waterNet.zero_grad()
        mark_face = Variable(img1).cuda()
        clean_face = Variable(img2).cuda()

        move_mark = waterNet(mark_face)
        fake_out = DNet(move_mark)

        move_mark_im1 = move_mark

        pix_loss = L2loss(move_mark_im1, clean_face)

        #feature_loss
        clean_im1 = img2
        clean_im_v = Variable(clean_im1).cuda()
        mark_im_v = move_mark
        g_feature_loss = feature_loss(clean_im_v, mark_im_v)

        #######################################################
        #local gan_losss
        local_face = img2[:, :, 25:95, 45:115]
        local_mark_face = move_mark.data[:, :, 25:95, 45:115]
        local_face_v = Variable(local_face).cuda()
        local_mark_face_v = Variable(local_mark_face)
        local_real_out = localNet(local_face_v)
        local_fake_out = localNet(local_mark_face_v)


        g_loss = beta1*fake_out + gamma * pix_loss + beta2*local_fake_out + 10*g_feature_loss
        g_loss.backward()
        g_optimizer.step()

        print('[%d/1000][%d/3187] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
          % (epoch, idx, loss.cpu().data[0].numpy(), g_loss.cpu().data[0].numpy(), real_out.cpu().data[0].numpy(), fake_out.cpu().data[0].numpy()))
        data_g.append(g_loss.cpu().data[0].numpy()[0])
        data_d.append(loss.cpu().data[0].numpy()[0])
        data_pix.append(pix_loss.cpu().data[0])
        #print(pix_loss.cpu().data[0])


    #if epoch % 20 == 0:
    test_v = Variable(test_img, volatile=True).cuda()
    test_result = waterNet(test_v)
    vutils.save_image(test_result.data.mul(0.5).add(0.5), 'out/image/fake_samples_{0}.png'.format(epoch))

    if epoch % 5 == 0:
        torch.save(waterNet.state_dict(), 'out/model/waterNet_epoch_{0}.pth'.format(epoch))
        torch.save(DNet.state_dict(), 'out/model/netD_epoch_{0}.pth'.format(epoch))
        torch.save(localNet.state_dict(), 'out/model/localNet_epoch_{0}.pth'.format(epoch))




















