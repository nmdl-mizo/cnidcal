import numpy as np
from numpy import dot, cross
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt
from cnidcal.cnidcal import reciprocal_cell, search_CSL

def get_R_to_screen(lattice):
    """
    get a rotation matrix to make the interface plane of the slab located in the screen
    """
    v2 = lattice[:,1]
    v2 = v2 / norm(v2)
    v1 = cross(lattice[:,1], lattice[:,2])
    v1 = v1 / norm(v1)
    v3 = cross(v1, v2)
    v3 = v3 / norm(v3)
    here = np.column_stack((v2,v3,v1))
    there = np.eye(3)
    return dot(there, inv(here))

def get_lattice_points(basis, lim):
    """
    get a series of lattice points of a basis with column vectors
    """
    x = np.arange(-lim, lim + 1, 1)
    y = x
    indice = (np.stack(np.meshgrid(x, y)).T).reshape(len(x) ** 2, 2)
    return dot(basis, indice.T).T

def get_2D_reciprocal_CSL(PB):
    """
    get the two-D reciprocal lattice basis from a two-D plane basis
    """
    normal = cross(PB[:,0], PB[:,1])
    L_3D = np.column_stack((normal, PB))
    L_reciprocal = reciprocal_cell(L_3D)
    return L_reciprocal.T[[1,2]].T

def Xs_Ys_cell(lattice):
    """
    get the coordinates of the four veticies of a cell
    """
    P1 = np.array([0,0,0])
    P2 = lattice[:,0]
    P3 = lattice[:,0] + lattice[:,1]
    P4 = lattice[:,1]

    P1 = [0,0]
    P2 = [P2[0],P2[1]]
    P3 = [P3[0],P3[1]]
    P4 = [P4[0],P4[1]]

    x1 = [P1[0],P2[0]]
    y1 = [P1[1],P2[1]]

    x2 = [P2[0],P3[0]]
    y2 = [P2[1],P3[1]]

    x3 = [P3[0],P4[0]]
    y3 = [P3[1],P4[1]]

    x4 = [P4[0],P1[0]]
    y4 = [P4[1],P1[1]]
    xs = [x1, x2, x3, x4]
    ys = [y1, y2, y3, y4]
    return xs, ys

class cellsdrawer:
    """
    A class saving information of plane bases and thier CSL and CNID
    with functions for visualization based on matplotlib
    """

    def __init__(self, PB_1, PB_2, CSL, CNID, lim):
        self.PB_1 = PB_1 # plane bases
        self.PB_2 = PB_2
        self.PB_1_R = get_2D_reciprocal_CSL(PB_1) # reciprocal plane bases
        self.PB_2_R = get_2D_reciprocal_CSL(PB_2)
        self.CSL = CSL  # 2-D CSL (direct)
        self.CSL_R = search_CSL(self.PB_1_R, self.PB_2_R, lim) # 2-D CSL (reciprocal)
        self.CNID = CNID # CNID
        
    def draw_direct(self, xlow, xhigh, ylow, yhigh, figsize_x, figsize_y, show_CSL = True, \
    show_CNID_points = False, show_CNID_cell = False, show_lattice_1 = True, show_lattice_2 = True, size_LP_1 = 150, \
    size_LP_2 = 50, save = False, dpi = 600, show_axis = False, lim = 50, show_legend = False, filename = 'Figure'):
        """
        A function drawing the direct lattices
        """
        
        #Rotate to face the screen
        normal = cross(self.PB_1[:,0],self.PB_1[:,1])
        R_to_screen = get_R_to_screen(np.column_stack((normal, self.PB_1)))
        self.PB_1_screen = dot(R_to_screen, self.PB_1)
        self.PB_2_screen = dot(R_to_screen, self.PB_2)
        self.CNID_screen = dot(R_to_screen, self.CNID)
        self.CSL_screen = dot(R_to_screen, self.CSL)
        LP1 = get_lattice_points(self.PB_1_screen,lim)
        LP2 = get_lattice_points(self.PB_2_screen,lim)
        LPCNID = get_lattice_points(self.CNID_screen, lim)
        #Draw
        plt.figure(figsize = (figsize_x,figsize_y))
        plt.scatter(LP1[:,0], LP1[:,1], s = size_LP_1, alpha = 0.3, c = 'b', label = 'Lattice 1')
        plt.scatter(LP2[:,0], LP2[:,1], s = size_LP_2, alpha = 0.7, c = 'orange', label = 'Lattice 2')
        if show_CNID_points == True:
            plt.scatter(LPCNID[:,0], LPCNID[:,1], s = 30, alpha = 0.5, c = 'g', label = 'CNID')
        if show_lattice_1 == True:
            xs, ys = Xs_Ys_cell(self.PB_1_screen)
            plt.plot(xs, ys, c = 'b', linewidth = 3)
        if show_lattice_2 == True:
            xs, ys = Xs_Ys_cell(self.PB_2_screen)
            plt.plot(xs, ys, c = 'orange', linewidth = 3)
        if show_CSL == True:
            xs, ys = Xs_Ys_cell(self.CSL_screen)
            plt.plot(xs, ys, c = 'k', linewidth = 1)
        if show_CNID_cell == True:
            xs, ys = Xs_Ys_cell(self.CNID_screen)
            plt.plot(xs, ys, c = 'g')
        plt.axis('scaled')
        plt.xlim(xlow,xhigh)
        plt.ylim(ylow,yhigh)
        if show_axis == False:
            plt.axis('off')
        if save == True:
            plt.savefig(filename, dpi = 600, format = 'jpg')
        if show_legend == True:
        	  plt.legend()
    
    def draw_reciprocal(self, xlow, xhigh, ylow, yhigh, figsize_x, figsize_y, show_CSL_R = True, \
    show_lattice_1_R = True, show_lattice_2_R = True, size_LP_1_R = 150, \
    size_LP_2_R = 50, save = False, dpi = 600, show_axis = False, lim = 50, show_legend = False, filename = 'Figure'):
        """
        A function drawing the direct lattices
        """
        
        #Rotate to face the screen
        normal = cross(self.PB_1[:,0], self.PB_1[:,1])
        R_to_screen = get_R_to_screen(np.column_stack((normal, self.PB_1)))
        self.PB_R_1_screen = dot(R_to_screen, self.PB_1_R)
        self.PB_R_2_screen = dot(R_to_screen, self.PB_2_R)
        self.CSL_R_screen = dot(R_to_screen, self.CSL_R)
        LP1_R = get_lattice_points(self.PB_R_1_screen, 50)
        LP2_R = get_lattice_points(self.PB_R_2_screen, 50)
        LPCSL_R = get_lattice_points(self.CSL_R_screen, 50)
        
        #Draw
        plt.figure(figsize = (figsize_x,figsize_y))
        plt.scatter(LP1_R[:,0], LP1_R[:,1], s = size_LP_1_R, alpha = 0.3, c = 'b', label = 'Reciprocal Lattice 1', marker = 'h')
        plt.scatter(LP2_R[:,0], LP2_R[:,1], s = size_LP_2_R, alpha = 0.7, c = 'orange', label = 'Reciprocal Lattice 2', marker = 'h')
        if show_lattice_1_R == True:
            xs, ys = Xs_Ys_cell(self.PB_R_1_screen)
            plt.plot(xs, ys, c = 'b', linewidth = 3)
        if show_lattice_1_R == True:
            xs, ys = Xs_Ys_cell(self.PB_R_2_screen)
            plt.plot(xs, ys, c = 'orange', linewidth = 3)
        if show_CSL_R == True:
            xs, ys = Xs_Ys_cell(self.CSL_R_screen)
            plt.plot(xs, ys, c = 'k', linewidth = 1)
        plt.axis('scaled')
        plt.xlim(xlow,xhigh)
        plt.ylim(ylow,yhigh)
        if show_axis == False:
            plt.axis('off')
        if save == True:
            plt.savefig(filename, dpi = 600, format = 'jpg')
        if show_legend == True:
        	  plt.legend()        
