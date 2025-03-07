
def downsample(pcd, factor=0.2):
    # Ensure the voxel size is optimally small for precision, but not too small to slow down the processing
    if pcd.is_empty():
        return pcd
    downsample_pcd = pcd.voxel_down_sample(voxel_size=factor)
    return downsample_pcd
