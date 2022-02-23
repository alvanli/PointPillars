import glob
import numpy as np
import open3d as o3d
from pypcd import pypcd

def visualize_pc(new_pc_rect, label=None):
    if np.random.random() < 1:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(new_pc_rect[:, :3])
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
        if label is not None:
            pc_colors = np.repeat([0, 0, 1], len(new_pc_rect)).reshape(3, len(new_pc_rect)).T
            pc_colors[np.argwhere(label==1)] = [1, 0, 0]
            print("pc_colors", pc_colors.shape)
            pcd.colors = o3d.utility.Vector3dVector(pc_colors)
        o3d.visualization.draw_geometries([pcd, axes])


if __name__ == "__main__":
    for filename in glob.iglob("/mnt/wato-drive/perception_pcds/feb2022/pc*.pcd", recursive=True):
        file_num = filename.split("/")[-1][2:].split(".")[0]
        pc = pypcd.PointCloud.from_path(filename)
        pc_data = pc.pc_data
        a = pc_data["intensity"]
        print("pcdata", a)
        pcd_in = o3d.io.read_point_cloud(filename).remove_non_finite_points(remove_nan=True)
        print(pcd_in.points.shape)
        input()
        np_pcd = np.asarray(pcd_in.points)
        print(len(np_pcd))
        np.save("/mnt/wato-drive/perception_pcds/road/npy/{}.npy".format(file_num), np_pcd)