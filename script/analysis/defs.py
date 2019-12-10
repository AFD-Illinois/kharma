# Definitions of enums and slices used throughout the code

from enum import Enum


class Met(Enum):
    """Enum of the metrics/coordinate systems supported by HARM"""
    MINKOWSKI = 0
    MKS = 1
    #MMKS = 2 # TODO put back support?
    FMKS = 3
    # Exotic metrics from KORAL et al
    EKS = 4
    MKS3 = 5
    # For conversions, etc
    KS = 6


class Loci(Enum):
    """Location enumerated value.
    Locations are defined by:
    ^ theta
    |----------------------
    |                     |
    |                     |
    |FACE1   CENT         |
    |                     |
    |CORN    FACE2        |
    -------------------------> R
    With FACE3 as the plane in phi"""
    FACE1 = 0
    FACE2 = 1
    FACE3 = 2
    CENT = 3
    CORN = 4


class Var(Enum):
    """All possible variables HARM supports. May not all be used in a given run"""
    RHO = 0
    UU = 1
    U1 = 2
    U2 = 3
    U3 = 4
    B1 = 5
    B2 = 6
    B3 = 7
    KTOT = 8
    KEL = 9


class Slices:
    """These slices can be constructed easily and define the bulk (all physical) fluid zones,
    separately from the ghost zones used for MPI syncing and boundary conditions

    Careful not to use the slices on arrays which are themselves slices of the whole! (TODO fix this requirement?)
    """

    def __init__(self, G):
        # Slices to represent variables, to add to below for picking out e.g. bulk of RHO
        self.allv = (slice(None),)
        self.RHO = (Var.RHO.value,)
        self.UU = (Var.UU.value,)
        self.U1 = (Var.U1.value,)
        self.U2 = (Var.U2.value,)
        self.U3 = (Var.U3.value,)
        self.B1 = (Var.B1.value,)
        self.B2 = (Var.B2.value,)
        self.B3 = (Var.B3.value,)
        self.KTOT = (Var.KTOT.value,)
        self.KEL = (Var.KEL.value,)
        # Single slices for putting together operations in bounds.py.  May be replaced by loopy kernels
        # Name single slices for character count
        ng = G.NG
        self.a = slice(None)
        self.b = slice(ng, -ng)
        self.bulk = (self.b, self.b, self.b)
        self.all = (slice(None),slice(None),slice(None))
        # "Halo" of 1 zone
        self.bh1 = slice(ng - 1, -ng + 1)
        self.bulkh1 = (self.bh1, self.bh1, self.bh1)

        # For manual finite-differencing.  Probably very slow
        self.diffr1 = (slice(ng + 1, -ng + 1), self.b, self.b)
        self.diffr2 = (self.b, slice(ng + 1, -ng + 1), self.b)
        self.diffr3 = (self.b, self.b, slice(ng + 1, -ng + 1))

        # Name boundaries slices for readability
        # Left side
        self.ghostl = slice(0, ng)
        self.boundl = slice(ng, 2*ng)
        self.boundl_r = slice(2 * ng, ng, -1)  # Reverse
        self.boundl_o = slice(ng, ng + 1)  # Outflow (1-zone slice for replication)
        # Right side
        self.ghostr = slice(-ng, None)
        self.boundr = slice(-2 * ng, -ng)
        self.boundr_r = slice(-ng, -2 * ng, -1)
        self.boundr_o = slice(-ng - 1, -ng)

    def geom_slc(self, slc):
        return slc[:2] + (None,)


class Shapes:
    def __init__(self, G):
        # Shapes for allocation
        self.geom_scalar = (G.GN[1], G.GN[2])
        self.geom_vector = (G.NDIM,) + self.geom_scalar
        self.geom_tensor = (G.NDIM,) + self.geom_vector

        self.grid_scalar = (G.GN[1], G.GN[2], G.GN[3])
        self.grid_vector = (G.NDIM,) + self.grid_scalar
        self.grid_tensor = (G.NDIM,) + self.grid_vector

        self.bulk_scalar = (G.N[1], G.N[2], G.N[3])
        self.bulk_vector = (G.NDIM,) + self.bulk_scalar
        self.bulk_tensor = (G.NDIM,) + self.bulk_vector
