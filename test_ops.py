from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


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


















