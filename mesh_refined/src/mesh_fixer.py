# sphinx_gallery_thumbnail_number = 1
import numpy as np
from pymeshfix import MeshFix
from pymeshfix._meshfix import PyTMesh
from pymeshfix.examples import planar_mesh
import pyvista as pv
import sys
import os

planar_mesh=sys.argv[1]
save_path=sys.argv[2]

##########################  加载mesh并展示其空洞 #######################################
#读取mesh
print(planar_mesh)
orig_mesh = pv.read(planar_mesh)

#计算mesh的孔洞
meshfix = MeshFix(orig_mesh)
holes = meshfix.extract_holes()
# meshfix.plot(show_holes=True)

print("start to fix mesh")
meshfix.repair(remove_smallest_components=True, joincomp=True)
mesh = meshfix.mesh
# meshfix.plot()

# Save the fixed mesh
# 在原始mesh的文件名后面加上_fixed
fixed_mesh = os.path.join(save_path, "fixed_" + planar_mesh.split('/')[-1])
print(fixed_mesh)
mesh.save(fixed_mesh)
