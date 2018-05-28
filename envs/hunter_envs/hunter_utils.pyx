def line_distance1(self, seg_a, seg_v_unit, seg_v_len, circ_pos, circ_rad):
    pt_v = [circ_pos[0] - seg_a[0], circ_pos[1] - seg_a[1]]
    proj = pt_v[0] * seg_v_unit[0] + pt_v[1] * seg_v_unit[1]
    if proj <= 0 or proj >= seg_v_len:
        return False
    proj_v = [seg_v_unit[0] * proj, seg_v_unit[1] * proj]
    closest = [int(proj_v[0] + seg_a[0]), int(proj_v[1] + seg_a[1])]
    dist_v = [circ_pos[0] - closest[0], circ_pos[1] - closest[1]]
    offset = sqrt(dist_v[0] ** 2 + dist_v[1] ** 2)
    if offset >= circ_rad:
        return False
    le = sqrt(circ_rad ** 2 - int(offset) ** 2)
    re = [closest[0] - seg_a[0], closest[1] - seg_a[1]]
    # if sqrt(re[0] ** 2 + re[1] ** 2) - le < 0:
    #     a = 1
    #     print a

    return sqrt(re[0] ** 2 + re[1] ** 2) - le