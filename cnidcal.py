"""
Here we provide codes to compute c.n.i.d, Mo and Or mentioned in the article.
Some codes are inspired by 'Hadian, Raheleh, Blazej Grabowski, and Jörg Neugebauer.
 "GB code: A grain boundary generation code." The Journal of Open Access Software 3 (2018)',
and Banadaki, Arash D., and Srikanth Patala. "An efficient algorithm for computing the primitive 
bases of a general lattice plane." Journal of Applied Crystallography 48.2 (2015): 585-588.
"""

import numpy as np
from numpy import dot, cross, square
from numpy.linalg import norm, inv


def get_fraction_basis(B):
    """
    out put an string of a basis with rational coefficients
    """
    B_c = np.empty_like(B)
    B_c[:, 0], n1 = find_integer_vectors(B[:, 0], 10000)
    B_c[:, 1], n2 = find_integer_vectors(B[:, 1], 10000)
    B_c = np.array(B_c, dtype=int)
    B_c = np.array(B_c, dtype=str)
    for i in range(3):
        B_c[:, 0][i] = B_c[:, 0][i] + "/" + str(n1)
        B_c[:, 1][i] = B_c[:, 1][i] + "/" + str(n2)
    return B_c


def get_coef_exp_ltc(lattice, carte):
    """
    convert a set of column vectors expressed in the cartesian coordinates
    to be expressed in a lattice coordinates
    arguments:
    lattice -- lattice basis
    carte -- column vectors cartesian coordinates
    """
    return dot(inv(lattice), carte)


def ang(a, b):
    """
    Returns value of cos(theta), where theta is the angle between vectors: a and b
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return dot(a, b) / (norm(a) * norm(b))


def rot(axis, theta):
    """Returns rotation matrix around axis with angle of theta in radian"""
    axis = np.array(axis) / norm(axis)
    return (
        np.cos(theta) * np.eye(3)
        + np.sin(theta) * np.array([
            [0., -axis[2], axis[1]],
            [axis[2], 0., -axis[0]],
            [-axis[1], axis[0], 0.]
        ])
        + (1.-np.cos(theta)) * np.outer(axis, axis)
    )


def signed_cell_volume(cell):
    """Returns signed cell volume by scalar triple product"""
    return dot(cell[:, 0], cross(cell[:, 1], cell[:, 2]))


def plane_cell_area(cell):
    """Returns signed cell volume by scalar triple product"""
    return norm(cross(cell[:, 1], cell[:, 2]))


def reciprocal_cell(cell):
    """Returns reciprocal cell (without factor of 2PI)"""
    v = signed_cell_volume(cell)
    recipro_cell = np.empty((3, 3))
    recipro_cell[:, 0] = cross(cell[:, 1], cell[:, 2]) / v
    recipro_cell[:, 1] = cross(cell[:, 2], cell[:, 0]) / v
    recipro_cell[:, 2] = cross(cell[:, 0], cell[:, 1]) / v
    return recipro_cell


def Basis(basis):
    """
    basis vectors of fcc and bcc lattice
    """
    if basis == 'fcc':
        return np.array([[0.5, 0, 0.5],
                         [0.5, 0.5, 0],
                         [0, 0.5, 0.5]], dtype=np.float64)
    elif basis == 'bcc':
        return np.array([[0.5, 0.5, 0.5],
                        [-0.5, -0.5, 0.5],
                        [0.5, -0.5, -0.5]], dtype=np.float64)
    else:
        ValueError('only available for fcc and bcc lattices to search the primitive CSL cell')


def find_integer_vectors(v, N_max, tol=1e-6):
    # A function find the natural number N up to N_max so that Nv contains only
    # integer elements with tolerance of tol and gcd(v1,v2,v3,N) = 1
    for N in range(1, N_max + 1):
        Nv = v * N
        if np.all(abs(Nv - np.round(Nv)) < tol):
            break
    else:
        raise RuntimeError('failed to find the rational vector of ' + str(v) + '\n within denominator <= ' + str(N_max))
    return np.round(Nv).astype(int), N


def projection(u1, u2):
    # get the projection of u1 on u2
    return dot(u1, u2) / dot(u2, u2)


def Gram_Schmidt(B0):
    # Gram–Schmidt process
    Bstar = np.eye(3, len(B0.T))
    for i in range(len(B0.T)):
        if i == 0:
            Bstar[:, i] = B0[:, i]
        else:
            BHere = B0[:, i].copy()
            for j in range(i):
                BHere = BHere - projection(B0[:, i], B0[:, j]) * B0[:, j]
            Bstar[:, i] = BHere
    return Bstar


def LLL(B):
    # LLL lattice reduction algorithm
    # https://en.wikipedia.org/wiki/Lenstra%E2%80%93Le-
    # nstra%E2%80%93Lov%C3%A1sz_lattice_basis_reduction_algorithm
    Bhere = B.copy()
    Bstar = Gram_Schmidt(Bhere)
    delta = 3 / 4
    k = 1
    while k <= len(B.T) - 1:
        js = -np.sort(-np.arange(0, k + 1 - 1))
        for j in js:
            ukj = projection(Bhere[:, k], Bstar[:, j])
            if abs(ukj) > 1 / 2:
                Bhere[:, k] = Bhere[:, k] - round(ukj) * Bhere[:, j]
                Bstar = Gram_Schmidt(Bhere)
        ukk_1 = projection(Bhere[:, k], Bstar[:, k - 1])
        if dot(Bstar[:, k], Bstar[:, k]) >= (delta - square(ukk_1)) * dot(Bstar[:, k - 1],Bstar[:, k - 1]):
            k += 1
        else:
            m = Bhere[:, k].copy()
            Bhere[:, k] = Bhere[:, k - 1].copy()
            Bhere[:, k - 1] = m
            Bstar = Gram_Schmidt(Bhere)
            k = max(k - 1, 1)
    return Bhere


def MID(lattice, n, tol = 1e-10, N_max=10000):
    # get the miller indices of a lattice plane with a normal vector n
    for i in range(3):
        if abs(dot(lattice[:, i], n)) > tol:
            Pc1 = lattice[:, i]
            break
    hkl = get_indices_from_n_Pc1(n, lattice, Pc1)
    hkl = find_integer_vectors(hkl, N_max)[0]
    return hkl


def get_primitive_hkl(hkl, C_lattice, P_lattice, N_max=10000):
    # convert the miller indices from conventional cell to primitive cell
    # 1. get normal
    n, Pc1 = get_plane(hkl, C_lattice)
    hkl_p = get_indices_from_n_Pc1(n, P_lattice, Pc1)
    hkl_p = find_integer_vectors(hkl_p, N_max)[0]
    return hkl_p


def get_plane(hkl, lattice):
    # get the normal vector and one in-plane point for the (hkl) plane of the lattice
    points = np.eye(3)
    for i in range(3):
        if hkl[i] != 0:
            points[:, 0] = lattice[:, i] / hkl[i]
            count = i
            break
    count2 = 1
    for i in range(3):
        if i != count:
            if hkl[i] == 0:
                points[:, count2] = points[:, 0] + lattice[:, i]
            else:
                points[:, count2] = lattice[:, i] / hkl[i]
            count2 += 1
    n = cross((points[:, 0] - points[:, 1]), (points[:, 0] - points[:, 2]))
    return n, points[:, 0]


def get_indices_from_n_Pc1(n, lattice, Pc1):
    # get the miller indices of certain plane with normal n
    # and one in-plane point Pc1 for certain lattice
    hkl = np.empty(3)
    for i in range(3):
        hkl[i] = dot(lattice[:, i], n) / dot(Pc1, n)
    return hkl


def ext_euclid(a, b):
    # extended euclidean algorithm
    # from https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
    old_s, s = 1, 0
    old_t, t = 0, 1
    old_r, r = a, b
    if b == 0:
        return 1, 0, a
    else:
        while(r != 0):
            q = old_r // r
            old_r, r = r, old_r - q * r
            old_s, s = s, old_s - q * s
            old_t, t = t, old_t - q * t
    return old_s, old_t, old_r


def get_pri_vec_inplane(hkl, lattice, N_max=1000):
    # get two primitive lattice vector
    h, k, l = hkl
    if k == 0 and l == 0:
        return LLL(np.column_stack((lattice[:, 1], lattice[:, 2])))
    by, bz, c = ext_euclid(abs(k), abs(l))
    if h == 0:
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, -l/c, k/c])
    else:
        bx = -c / h
        if k != 0:
            by = k / abs(k) * by
        if l != 0:
            bz = l / abs(l) * bz
        v1 = np.array([0, -l / c, k / c])
        v2 = np.array([bx, by, bz])
    v2 = find_integer_vectors(v2, N_max)[0]
    return LLL(dot(lattice, np.column_stack((v1, v2))))


def get_two_bases(lattice_1, lattice_2, R, miller_indices):
    """
    get the two plane bases in a coincident plane of lattice 1 & lattice 2
    arguments:
    lattice_1 --- column vectors of lattice 1
    lattice_2
    R --- rotation matrix applied on lattice 2
    miller_indices --- expressing the GB plane in lattice 1 frame
    return:
    B1, B2 --- two plane bases
    """
    PB_1 = get_pri_vec_inplane(miller_indices, lattice_1)
    lattice_2 = dot(R, lattice_2)
    normal = cross(PB_1[:, 0], PB_1[:, 1])
    miller_indices_2 = MID(lattice_2, normal)
    PB_2 = get_pri_vec_inplane(miller_indices_2, lattice_2)
    return PB_1, PB_2


def search_CSL(B1, B2, lim, ortho = False, tol = 1e-4):
    """
    finds CSL of two 2-D lattices
    arguments:
    B1 -- 3D basis
    B2 -- plane basis
    lim -- control num of generated reciprocal lattice points to find CSL
    ortho -- whether to obtain two orthogonal vectors
    tol -- tolerance to judge whether the two vectors are orthogonal
    return:
    v1, v2 -- two basic vectors of CSL
    """
    B1 = np.column_stack((cross(B1[:, 0], B1[:, 1]), B1))
    basis1 = B1.T
    basis2 = B2.T

    # meshes
    x = np.arange(-lim, lim + 1, 1)
    y = x

    indice = (np.stack(np.meshgrid(x, y)).T).reshape(len(x) ** 2, 2)
    indice_0 = indice[np.where(np.sum(abs(indice), axis=1) != 0)]
    indice_0 = np.array(indice_0, dtype=np.float64)
    # Reciprocal lattice points in the GB plane
    LP2 = dot(indice_0, basis2)

    # describe lattice-2 in coordinate-1
    L2f = dot(LP2, inv(basis1))
    # get CSL points
    L2f = L2f[np.where(np.all(abs(np.round(L2f) - L2f) < 1e-7, axis=1))[0]]

    # convert to cartesian coordinate
    L2f = dot(L2f, basis1)
    L2f = L2f[np.argsort(norm(L2f, axis=1))]

    # get the two minimum non linear vectors
    v1 = L2f[0]
    for i in L2f:
        if ortho == False:
            if (1 - abs(ang(i, v1)) > 1e-7):
                v2 = i
                found = True
                break
        else:
            if abs(dot(i, v1)) < tol:
                v2 = i
                found = True
                break
                
    else:
        raise RuntimeError('failed to find the csl of the reciprocal lattices in the gb plane, \
        maybe the inputted two lattices do not form a CSL, or please increase lim')

    return LLL(np.column_stack((v1, v2)))


def searchcnid(B1, B2, lim):
    """
    finds c.n.i.d 
    For more details of c.n.i.d,
    refer the book: 'Interfaces in crystalline materials',
     Sutton and Balluffi, clarendon press, 1996.
    arguments:
    B1 -- plane basis 1
    B2 -- plane basis 2
    lim -- control num of generated reciprocal lattice points to find CSL
    return:
    B -- two column vectors of CNID
    """
    # normal vector
    v_n = cross(B1[:, 0], B1[:, 1])
    RP1 = np.column_stack((v_n, B1))
    RP2 = np.column_stack((v_n, B2))

    # Compute the reciprocal Lattices
    R1 = reciprocal_cell(RP1)
    R2 = reciprocal_cell(RP2)

    # get the CSL of the 2D reciprocal lattice
    basis1 = np.column_stack((R1[:, 1], R1[:, 2]))
    basis2 = np.column_stack((R2[:, 1], R2[:, 2]))

    v1, v2 = search_CSL(basis1, basis2, lim).T

    cnidP = np.empty((3, 3))
    cnidP[:, 0] = R1[:, 0]
    cnidP[:, 1] = v1
    cnidP[:, 2] = v2
    # convert to direct lattice
    cnid = reciprocal_cell(cnidP)
    if norm(cnid[:, 1]) > norm(cnid[:, 2]):
        copyv = cnid[:, 1].copy()
        cnid[:, 1] = cnid[:, 2]
        cnid[:, 2] = copyv
    if signed_cell_volume(cnid) < 0:
        cnid[:, 0] = -cnid[:, 0]

    return LLL(np.column_stack((cnid[:, 1], cnid[:, 2])))

