from libc.math cimport sqrt
import cython
from cpython.array cimport array

@cython.boundscheck(False)
@cython.wraparound(False)
def line_distance_fast(int seg_a_0, int seg_a_1, double seg_v_unit_0, double seg_v_unit_1, int seg_v_len,
                       int circ_pos_0,
                       int circ_pos_1, int circ_rad):
    cdef int pt_v_0 = circ_pos_0 - seg_a_0
    cdef int pt_v_1 = circ_pos_1 - seg_a_1

    proj = pt_v_0 * seg_v_unit_0 + pt_v_1 * seg_v_unit_1
    if proj <= 0 or proj >= seg_v_len:
        return False
    cdef double proj_v_0 = seg_v_unit_0 * proj
    cdef double proj_v_1 = seg_v_unit_1 * proj

    cdef double closest_0 = proj_v_0 + seg_a_0
    cdef double closest_1 = proj_v_1 + seg_a_1

    cdef double dist_v_0 = circ_pos_0 - closest_0
    cdef double dist_v_1 = circ_pos_1 - closest_1

    offset = sqrt(dist_v_0 ** 2 + dist_v_1 ** 2)
    if offset >= circ_rad:
        return False

    cdef double le = sqrt(circ_rad ** 2 - offset ** 2)
    cdef double re_0 = closest_0 - seg_a_0
    cdef double re_1 = closest_1 - seg_a_1

    return sqrt(re_0 ** 2 + re_1 ** 2) - le

@cython.boundscheck(False)
@cython.wraparound(False)
def count_distance_fast(int a1, int a2, int b1, int b2):
    return sqrt((a1 - b1) ** 2 + (a2 - b2) ** 2)
