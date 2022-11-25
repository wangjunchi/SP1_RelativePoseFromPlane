import pyransac
import torch
import torchvision.transforms
from torch.utils.data import DataLoader

from PlaneDetection.models import Model
from PlaneDetection.Utilities.Visualize import Visualisation as vs
from PlaneDetection.Utilities.util import Utility
from PIL import Image
from tqdm import tqdm
from Dataset.hypersim_image_pair_dataset import HypersimImagePairDataset


def load_plane_detector(model_path):
    model = Model.Model(num_features=2048, block_channel=[256, 512, 1024, 2048], pretrained=None)
    model = torch.nn.DataParallel(model).cuda()

    state_dict = torch.load(model_path)['state_dict']
    model.load_state_dict(state_dict)

    return model

def predict_image(image, model):
    # model = Model.Model(num_features=2048, block_channel=[256, 512, 1024, 2048], pretrained=None)
    # model = torch.nn.DataParallel(model).cuda()
    #
    # weights_path = "/home/junchi/sp1/project_junchi/pythonProject/whole_pipeline/PlaneDetection/trained_models/pec_junchi.tar"
    # state_dict = torch.load(weights_path)['state_dict']
    # model.load_state_dict(state_dict)

    # check if image is tensor
    if not isinstance(image, torch.Tensor):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(228),
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(228),
        ])
    ransac_params = pyransac.RansacParams(samples=3, iterations=2, confidence=0.98, threshold=0.5)

    with torch.no_grad():
        model.eval()
        model.cuda()

        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = transform(image)

        output_plane_para, output_embedding = model(image)
        output_normal = output_plane_para / torch.norm(output_plane_para, dim=1, keepdim=True)

        labels = Utility.embedding_segmentation(output_embedding)
        depth = torch.norm(output_plane_para, dim=1, keepdim=True)
        normals = Utility.aggregate_parameters_ransac(labels, output_normal, ransac_params)

        return labels, depth, normals

if __name__ == "__main__":
    dataset = HypersimImagePairDataset(data_dir="/home/junchi/sp1/dataset/hypersim", scene_name='ai_001_001')

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for step, batch in tqdm(enumerate(loader)):
        sample_1 = batch['sample_1']
        sample_2 = batch['sample_2']
        gt_match = batch['gt_match']

        image_1 = sample_1['image']
        image_2 = sample_2['image']

        # predict planes
        labels_1, depth_1, normals_1 = predict_image(image_1)
        labels_2, depth_2, normals_2 = predict_image(image_2)

        # visualize
        vs.image_show(image_1, "Original image 1")
        vs.image_show(labels_1, "Plane segmentation 1")

        vs.image_show(image_2, "Original image 2")
        vs.image_show(labels_2, "Plane segmentation 2")

        pass
