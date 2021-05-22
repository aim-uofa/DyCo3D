
import numpy as np
def draw_line(xyz1, xyz2, num=1000):
    x1 = xyz1[0]
    x2 = xyz2[0]

    y1 = xyz1[1]
    y2 = xyz2[1]

    z1 = xyz1[2]
    z2 = xyz2[2]

    x_line = np.linspace(x1, x2, num).reshape([-1, 1])
    y_line = np.linspace(y1, y2, num).reshape([-1, 1])
    z_line = np.linspace(z1, z2, num).reshape([-1, 1])
    return np.concatenate([x_line, y_line, z_line], axis=-1)

def draw_3d_box_pcds(center, lx, ly, lz, pointnum):
    """
    Args:
        center: x, y, z
        lx:
        ly:
        lz:
    Returns:
    """

    pt0 = np.array([center[0] - lx / 2.0, center[1] - ly / 2.0, center[2] - lz / 2.0])
    pt1 = np.array([center[0] + lx / 2.0, center[1] - ly / 2.0, center[2] - lz / 2.0])
    pt2 = np.array([center[0] + lx / 2.0, center[1] - ly / 2.0, center[2] + lz / 2.0])
    pt3 = np.array([center[0] - lx / 2.0, center[1] - ly / 2.0, center[2] + lz / 2.0])

    pt4 = np.array([center[0] - lx / 2.0, center[1] + ly / 2.0, center[2] - lz / 2.0])
    pt5 = np.array([center[0] + lx / 2.0, center[1] + ly / 2.0, center[2] - lz / 2.0])
    pt6 = np.array([center[0] + lx / 2.0, center[1] + ly / 2.0, center[2] + lz / 2.0])
    pt7 = np.array([center[0] - lx / 2.0, center[1] + ly / 2.0, center[2] + lz / 2.0])

    line0 = draw_line(pt0, pt1, pointnum)
    line1 = draw_line(pt1, pt2, pointnum)
    line2 = draw_line(pt2, pt3, pointnum)
    line3 = draw_line(pt3, pt0, pointnum)

    line4 = draw_line(pt4, pt5, pointnum)
    line5 = draw_line(pt5, pt6, pointnum)
    line6 = draw_line(pt6, pt7, pointnum)
    line7 = draw_line(pt7, pt4, pointnum)

    line8 = draw_line(pt0, pt4, pointnum)
    line9 = draw_line(pt1, pt5, pointnum)
    line10 = draw_line(pt2, pt6, pointnum)
    line11 = draw_line(pt3, pt7, pointnum)

    return np.concatenate([line0, line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11], axis=0)


def draw_3d_box_pcds2(center, lx1, lx2, ly1, ly2, lz1, lz2, pointnum):
    xmin = center[0] - lx1
    xmax = center[0] + lx2

    ymin = center[1] - ly1
    ymax = center[1] + ly2

    zmin = center[2] - lz1
    zmax = center[2] + lz2

    center = [(xmin+xmax)/2.0, (ymin+ymax)/2.0, (zmin+zmax)/2.0]
    return draw_3d_box_pcds(center=center, lx=lx1+lx2, ly=ly1+ly2, lz=lz1+lz2, pointnum=pointnum)




def write_ply_color(points, labels, out_filename, num_classes=None):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    import matplotlib.pyplot as pyplot
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels)+1
    else:
        assert(num_classes>np.max(labels))
    fout = open(out_filename, 'w')
    colors = [pyplot.cm.hsv(i/float(num_classes)) for i in range(num_classes)]
    for i in range(N):
        c = colors[labels[i]]
        c = [int(x*255) for x in c]
        fout.write('v %f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]))
    fout.close()

def write_ply_rgb(points, colors, out_filename):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as OBJ file """
    colors = colors.astype(int)
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i, :]
        fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()