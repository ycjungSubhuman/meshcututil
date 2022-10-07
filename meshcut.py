import numpy as np
import igl
from scipy.spatial import KDTree

class MeshCut:
    """
    Given a mesh and its cut example, cut any given mesh in the same manner.
    """
    def __init__(self, path_orig, path_cut):
        """
        path_orig      path to the original mesh (.obj, .ply)
        path_cut       path to the cut example (.obj, .ply)

        The mesh and the cut example must be aligned.
        """
        self.mesh_orig = igl.read_triangle_mesh(path_orig)
        self.mesh_cut = igl.read_triangle_mesh(path_cut)

        self._build_mapping()

    def _build_mapping(self):
        tree = KDTree(self.mesh_orig[0])
        _, I = tree.query(self.mesh_cut[0])
        self.I = I # V_orig[I] == V_cut
        self.Iinv = -np.ones(self.mesh_orig[0].shape[0], dtype=np.int32)
        self.Iinv[I] = np.arange(self.mesh_cut[0].shape[0], dtype=np.int32)

    def cut(self, V):
        """
        Cut an arbitrary mesh in the topology of mesh_orig.

        V                (N, m) vertices of the mesh to be cut
                         E.g., position array or UV array
        """
        return V[self.I]
        
    def convert_index_cut_to_orig(self, l_cut):
        """
        Convert landmark vertex indices of cut mesh to landmark indices of original mesh

        l_cut           (N,) landmark vertex indices of cut mesh
        """
        return self.I[l_cut]
        
    def convert_index_orig_to_cut(self, l_orig):
        """
        Convert landmark vertex indices of original mesh to landmark indices of the cut mesh
        If not found, the vertex index is marked with -1

        l_orig          (N,) landmark vertex indices of original mesh
        """
        return self.Iinv[l_orig]

    def faces(self):
        """
        Face indeces for the cut mesh

        Returns
        (#F, 3) face indices
        """
        return self.mesh_cut[1]