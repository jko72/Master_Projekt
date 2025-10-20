import yaml
import torch
import sys
import random

from utils.projection import projection
from pyTorchChamferDistance.chamfer_distance import ChamferDistance

device = torch.device('cuda')

class cham_dist(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss = ChamferDistance()
        self.projection = projection(self.cfg)

    def forward(self, output, target, n_samples = -1):
        batch_size, n_future_steps, H, W = output["rv"].shape

        # Getting output point cloud
        masked_output = self.projection.get_masked_range_view(output)
        output_points = self.projection.get_valid_points_from_range_view(
                    masked_output.view(batch_size*n_future_steps, H,W), use_batch=True
                )
        output_points = output_points.view(batch_size*n_future_steps, -1, 3)

        # Getting target point cloud
        #print("Target shape: ", target.shape)
        target_points = target[:, :, 1:4, :, :].permute(0,1,3,4,2)
        target_points[target[:, :, 0, :, :] < 0.0] = torch.tensor([1e3 ,1e3 ,1e3], dtype=torch.float32).to(device)
        target_points = target_points.contiguous().view(batch_size*n_future_steps, -1, 3)

        # Padding with infinity 
        padding_tensor = torch.tensor([[[1e3, 1e3, 1e3]]]).to(device)
        padding_tensor = padding_tensor.repeat(batch_size*n_future_steps, 1, 1)
        output_points = torch.cat([output_points, padding_tensor], dim=1)
        target_points = torch.cat([target_points, padding_tensor], dim=1)

        # Calculating CD
        dist1, dist2 = self.loss(output_points, target_points)

        # Identifying valid distances
        mask1 = dist1 > 0
        mask2 = dist2 > 0
        positive1 = torch.sum(mask1, dim=1)
        positive2 = torch.sum(mask2, dim=1)

        # Arranging in tensor format
        dist_combined = torch.sum(dist1, dim=1)/positive1 + torch.sum(dist2, dim=1)/positive2 # Shape B*S, 1
        chamfer_distances_tensor = dist_combined.view(n_future_steps, batch_size)
        chamf_dist_t = torch.mean(chamfer_distances_tensor, dim=1)
        chamfer_distances = {i: chamf_dist_t[i].to('cpu') for i in range(len(chamf_dist_t))}
        return chamfer_distances, chamfer_distances_tensor.to('cpu')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_filename = "/csehome/aphale.1/PCPNet/config/parameters.yaml"
    cfg = yaml.safe_load(open(config_filename))

    #cd = chamfer_distance(cfg)

    cd = None

    rand1 = torch.randn(2,5,64,2048).to(device)
    rand2 = torch.randn(2,4,5,64,2048).to(device)
    rand3 = torch.randn(2,5,64,2048).to(device)

    output = {}
    output['rv'] = rand1
    output['mask_logits'] = rand3
    target = rand2

    chamfer_dist, chamfer_dist_tensor = cd(
                output, target, -1
            )
    
    print("Chamfer Dist using method 1 = ", chamfer_dist_tensor)

    CD = cham_dist(cfg)
    chamfer_dist, chamfer_dist_tensor = CD(
                output, target
            )
    
    print("Chamfer Dist using method 2 = ", chamfer_dist_tensor)