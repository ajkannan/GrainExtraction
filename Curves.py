""" Finds, returns, and plots the curves in a labeled pixel image.

Determines the curves by parsing the image provided cell by cell once.
Curves include the border curves, which have the highest ids. Curves end
at junctions, when they intersect boundary curves, or when they intersect
with themselves.

Last updated: 1/4/12 """

from numpy import ndarray as np
import matplotlib.pyplot as mpl
import collections as c
import Grains

def find_curves	(image):
    """ Iterates through the picture once to gather curves and grains.

    Iterates cell by cell, adding segments together until contour curves are
    finished.  Scales points on curves, plots them, and then returns a list
    of lists of tuples representing coordinates.  Calls find_grains() to obtain
    list of grains in the image.

    Arguments:
        image - rectangular image with labeled pixels

    Returns:
        List with two items:
            A list of curve pts comprised of lists of scaled curve pts
            A list of grains pts comprised of lists of scaled grain pts
            Every two consecutive lists within the list of curve and grain pts
                represent x and y coordinates """

    curves = {} # {curve_id: list containing 2 deques representing x and y pts}
    unconnected_pts = {} # Endpts of unfinished curves {(i,j): curve_id}
    next_curve_id = 0 # Next open keys for insertion of new curves
    junctions = {} # {junction pt: set of curve_ids}
    boundary_pts = set() # Set of boundary points

    # Traverses picture once to trace curves
    for i in xrange(image.shape[0] - 1):
        for j in xrange(image.shape[1] - 1):
            num_labels = count_labels(i, j, image)
            pts_to_add = check_cell(i, j, image)
            if len(pts_to_add) == 4:
                if num_labels < 4: # Arbitrary diagonal case
                    next_curve_id = append_segment(pts_to_add[0], pts_to_add[3],
                                                   unconnected_pts, curves,
                                                   next_curve_id, image.shape,
                                                   junctions, boundary_pts)
                    next_curve_id = append_segment(pts_to_add[1], pts_to_add[2],
                                                   unconnected_pts, curves,
                                                   next_curve_id, image.shape,
                                                   junctions, boundary_pts)
                else: # 4-way junction
                    junctions[(i + 0.5, j + 0.5)] = set()
                    for current_pt in pts_to_add:
                        next_curve_id = append_segment(current_pt,
                                                       (i + 0.5, j + 0.5),
                                                       unconnected_pts, curves,
                                                       next_curve_id,
                                                       image.shape, junctions,
                                                       boundary_pts)
            elif num_labels == 3: # 3-way junction
                junctions[(i + 0.5, j + 0.5)] = set()
                for current_pt in pts_to_add:
                    next_curve_id = append_segment(current_pt,
                                                   (i + 0.5, j + 0.5),
                                                   unconnected_pts, curves,
                                                   next_curve_id, image.shape,
                                                   junctions, boundary_pts)
            elif num_labels == 2: # Non-junction contour
                next_curve_id = append_segment(pts_to_add[0], pts_to_add[1],
                                               unconnected_pts, curves,
                                               next_curve_id, image.shape,
                                               junctions, boundary_pts)

    # Find boundary curves and boundary curve ids
    boundary_curves = find_boundaries(curves, junctions, boundary_pts,
                                      image.shape, next_curve_id)
    # Find grains
    grains = Grains.find_grains(curves, boundary_curves, junctions)

    # Scale, plot and return list of curves
    scale(curves, image.shape)
    scale(grains, image.shape)

    mpl.figure()
    for curve_id in curves.keys():
        mpl.plot(curves[curve_id][0], curves[curve_id][1], 'o-')
    return [curves.values(), grains.values()]


def append_segment(pt1, pt2, unconnected_pts, curves, next_curve_id, dimensions,
                   junctions, boundary_pts):
    """ Appends segments of contour lines to the appropriate curves.

    Given two points that represent a segment to add to a contour curve,
    append_pt figures out to which existing curves the segment should be added
    (if any).  Starts a new curve in the dictionary curves if no match found.

    Arguments:
        pt1 - first point in segment of curve
        pt2 - second point in segment of curve
        unconnected_pts - dictionary (keys are endpoints of contour curves that
            are not yet finished, values are keys of corresponding curves
        curves - dictionary (keys are arbitrary integers, values are lists of
            lists that represents points on contour curves
        next_curve_id - the next available id for a new curve in the dictionary
        dimensions - tuple representing dimensions of image (row, col)
        junctions - a dictionary of junction pts, values being sets of connected
            curve ids
        boundary_pts - a set of ids that represent boundary curve pieces

    Returns:
        Integer representing a key available for the next new curve """

    # Adds junctions at boundary points
    if pt1 not in boundary_pts and is_boundary_pt(pt1, dimensions):
        junctions[pt1] = set()
        boundary_pts.add(pt1)
    if pt2 not in boundary_pts and is_boundary_pt(pt2, dimensions):
        junctions[pt2] = set()
        boundary_pts.add(pt2)

    # Case 1: Segment added will adjoin two curves
    if pt1 in unconnected_pts and pt2 in unconnected_pts:
        adjoin_curves(pt1, pt2, unconnected_pts, curves, junctions, boundary_pts)

    # Cases 2 & 3: Segment should be appended to an existing curve
    elif bool(pt1 in unconnected_pts) ^ bool(pt2 in unconnected_pts):
        end_pt = None
        new_pt = None
        curve_id = None
        # Case 2
        if pt1 in unconnected_pts:
            end_pt = pt1
            new_pt = pt2
            curve_id = unconnected_pts[pt1]
        # Case 3
        else:
            end_pt = pt2
            new_pt = pt1
            curve_id = unconnected_pts[pt2]

        # Updates junctions if pt1 is on boundary
        if pt1 in junctions:
            junctions[pt1].add(curve_id)
        # Updates junctions if pt2 is on boundary or is a junction
        elif pt2 in junctions:
            junctions[pt2].add(curve_id)

        # Checks which end of the curve the point should be added and checks
        # for extra points on the curve
        if (end_pt[0] == curves[curve_id][0][0] and end_pt[1] ==
            curves[curve_id][1][0]):
            if not replace_pt(pt1, pt2, curves, curve_id, 1, dimensions):
                curves[curve_id][0].appendleft(new_pt[0])
                curves[curve_id][1].appendleft(new_pt[1])
        else:
            if not replace_pt(pt1, pt2, curves, curve_id, -2, dimensions):
                curves[curve_id][0].append(new_pt[0])
                curves[curve_id][1].append(new_pt[1])

        # Adjust endpoint in unconnected_pts if necessary
        if not is_boundary_pt(new_pt, dimensions) and not is_junction_pt(new_pt):
            unconnected_pts[new_pt] = unconnected_pts[end_pt]
        del unconnected_pts[end_pt]

    # Case 4: Segment is not connected to any other curve
    else:
        curves[next_curve_id] = [c.deque((pt1[0], pt2[0])),
                                 c.deque((pt1[1], pt2[1]))]
        # Check that point is not a boundary or junction before insertion
        if not is_boundary_pt(pt1, dimensions) and not is_junction_pt(pt1):
            unconnected_pts[pt1] = next_curve_id
        else:
            junctions[pt1].add(next_curve_id)

        if not is_boundary_pt(pt2, dimensions) and not is_junction_pt(pt2):
            unconnected_pts[pt2] = next_curve_id
        else:
            junctions[pt2].add(next_curve_id)

        next_curve_id += 1 # Change next availabe curve_id
    return next_curve_id


def adjoin_curves(pt1, pt2, unconnected_pts, curves, junctions, boundary_pts):
    """ Finishes closed curves or adjoins two different curves.

    Given two points, both of which are connected to curve[s], this function
    decides how to incorporate the segment represented by pt1 and pt2.  If both
    curves are connected to the same curve, then the curve is a closed curve.
    If not, the two different curves are added together, keeping in mind the
    orientations of the curves.

    Arguments:
        pt1 - first point in segment of curve
        pt2 - second point in segment of curve
        unconnected_pts - dictionary (keys are endpoints of contour curves that
            are not yet finished, values are keys of corresponding curves
        curves - dictionary (keys are arbitrary integers, values are lists of
            lists that represents points on contour curves
        junctions - a dictionary of junctions pt keys and set values of connected
            curve ids
        boundary_pts - a set of ids that represents boundary curve pieces

    Returns nothing """

    curve1_id = unconnected_pts[pt1]
    curve2_id = unconnected_pts[pt2]

    if curve1_id == curve2_id: # Closed curve, close the loop
        # pt1 is at beginning of curve
        if (pt1[0] == curves[curve1_id][0][0] and pt1[1] ==
            curves[curve1_id][1][0]):
            rm_adjoining_pts(pt1, pt2, curve1_id, 1, curve1_id, -2, curves)
        else:
            rm_adjoining_pts(pt1, pt2, curve1_id, -2, curve1_id, 1, curves)
        curves[curve1_id][0].append(curves[curve1_id][0][0])
        curves[curve1_id][1].append(curves[curve1_id][1][0])

    else: # Two different curves must be adjoined
        if (pt1[0] == curves[curve1_id][0][0] and pt1[1] ==
            curves[curve1_id][1][0]):
            # Both points at beginning
            # Reverse curve 1 and append to beginning of curve 2
            if (pt2[0] == curves[curve2_id][0][0] and pt2[1] ==
                curves[curve2_id][1][0]):
                rm_adjoining_pts(pt1, pt2, curve1_id, 1, curve2_id, 1, curves)
                curves[curve2_id][0].extendleft(curves[curve1_id][0])
                curves[curve2_id][1].extendleft(curves[curve1_id][1])

            # Points are on opposite ends, add curve 1 to the end of curve 2
            else:
                rm_adjoining_pts(pt1, pt2, curve1_id, 1, curve2_id, -2, curves)
                curves[curve2_id][0].extend(curves[curve1_id][0])
                curves[curve2_id][1].extend(curves[curve1_id][1])

            # Delete curve 1 and change id in unconnected_pts if necessary
            c1_last_pt = (curves[curve1_id][0][-1], curves[curve1_id][1][-1])
            if c1_last_pt in unconnected_pts:
                unconnected_pts[c1_last_pt] = curve2_id
            if c1_last_pt in junctions:
                       junctions[c1_last_pt].remove(curve1_id)
                       junctions[c1_last_pt].add(curve2_id)
            del curves[curve1_id]

        else:
            # Both points are at end, curve 2 needs to be reversed
            if (pt2[0] == curves[curve2_id][0][-1] and pt2[1] ==
                curves[curve2_id][1][-1]):
                curves[curve2_id][0] = c.deque(reversed(curves[curve2_id][0]))
                curves[curve2_id][1] = c.deque(reversed(curves[curve2_id][1]))
            # Add curve 2 to end of curve 1
            rm_adjoining_pts(pt1, pt2, curve1_id, -2, curve2_id, 1, curves)
            curves[curve1_id][0].extend(curves[curve2_id][0])
            curves[curve1_id][1].extend(curves[curve2_id][1])

            # Delete curve 2 and change id in unconnected_pts if necessary
            c2_last_pt = (curves[curve2_id][0][-1], curves[curve2_id][1][-1])
            if ((curves[curve2_id][0][-1], curves[curve2_id][1][-1]) in
                unconnected_pts):
                unconnected_pts[c2_last_pt] = curve1_id
            # Updates junctions for changed curve_id
            if c2_last_pt in junctions:
                junctions[c2_last_pt].remove(curve2_id)
                junctions[c2_last_pt].add(curve1_id)
            del curves[curve2_id]
    del unconnected_pts[pt1]
    del unconnected_pts[pt2]


def find_boundaries(curves, junctions, boundary_pts, dimensions, next_curve_id):
    """Adds the boundary curves to the dictionary of curves.

    Determines the boundary curve pieces in an image.  Iterates through the
    boundary points from the top left corner (image index 0, 0), clockwise.
    When the pixel label changes, a new boundary curve is added.

    Arguments:
        curves - dictionary (keys are arbitrary integers, values are lists of
            lists that represents points on contour curves
        junctions - a dictionary of junctions pt keys and set values of connected
            curve ids
        boundary_pts - a set of ids that represents boundary curve pieces
        next_curve_id - the next available id for a new curve in the dictionary
        dimensions - tuple representing dimensions of image (row, col)

    Returns:
        A set of curve_ids for boundary curves """

    init_id = next_curve_id
    current_curve = [c.deque((0,)),c.deque((0,))]
    boundary_curves = set() # set of curve_ids for boundary curves
    # Top edge
    for j in xrange(dimensions[1] - 1):
        if (0, j + 0.5) in boundary_pts:
            current_curve = add_b_curve(curves, current_curve, junctions,
                                        (0, j + 0.5), next_curve_id,
                                        boundary_curves)
            next_curve_id += 1
        if j == dimensions[1] - 2: # Add top right corner
            current_curve[0].append(0)
            current_curve[1].append(j + 1)

    # Right edge
    for i in xrange(dimensions[0] - 1):
        if (i + 0.5, dimensions[1] - 1) in boundary_pts:
            current_curve = add_b_curve(curves, current_curve, junctions,
                                        (i + 0.5, dimensions[1] - 1),
                                        next_curve_id, boundary_curves)
            next_curve_id += 1
        if i == dimensions[0] - 2: # Add bottom right corner
            current_curve[0].append(i + 1)
            current_curve[1].append(dimensions[1] - 1)

    # Bottom edge
    for j in xrange(dimensions[1] - 1, 0, -1):
        if (dimensions[0] - 1, j - 0.5) in boundary_pts:
            current_curve = add_b_curve(curves, current_curve, junctions,
                                        (dimensions[0] - 1, j - 0.5),
                                        next_curve_id, boundary_curves)
            next_curve_id += 1
        if j == 1: # Add bottom left corner
            current_curve[0].append(dimensions[0] - 1)
            current_curve[1].append(0)

    # Left edge
    for i in xrange(dimensions[0] - 1, 0, -1):
        if (i - 0.5, 0) in boundary_pts:
            current_curve = add_b_curve(curves, current_curve, junctions,
                                        (i - 0.5, 0), next_curve_id,
                                        boundary_curves)
            next_curve_id += 1
        if i == 1 and init_id == next_curve_id: # Add top left corner
            current_curve[0].append(0)
            current_curve[1].append(0)

    # If the border is never touched, the current curve is added to curves
    if next_curve_id == init_id:
        curves[init_id] = current_curve
        boundary_curves.add(init_id)
    # Handles connection at top left corner between piece on top edge and piece
    # on left edge
    else:
        # Adds the initial boundary curve to the end of the last boundary curve
        curves[next_curve_id] = current_curve
        curves[next_curve_id][0].extend(curves[init_id][0])
        curves[next_curve_id][1].extend(curves[init_id][1])
        # Update curve_id in other boundary_curves and junctions
        boundary_curves.remove(init_id)
        boundary_curves.add(next_curve_id)
        junction_index = (curves[init_id][0][-1], curves[init_id][1][-1])
        junctions[junction_index].remove(init_id)
        junctions[junction_index].add(next_curve_id)
        del curves[init_id]
    return boundary_curves


# Adds a boundary curve to the list of boundary curves
def add_b_curve(curves, current_curve, junctions, pt, next_curve_id,
                boundary_curves):
    # Adds the final point to the curve and then adds the curve to the dictionary
    current_curve[0].append(pt[0])
    current_curve[1].append(pt[1])
    curves[next_curve_id] = current_curve
    # Records the junction between the boundary curves
    junctions[pt].add(next_curve_id)
    junctions[pt].add(next_curve_id + 1)
    boundary_curves.add(next_curve_id)
    return [c.deque((pt[0],)), c.deque((pt[1],))] # Starts a new curve


def replace_pt(pt1, pt2, curves, curve_id, pt_index, dimensions):
    """ Removes all but the endpoints of any straight line when extending curves.

    Checks for horizontal, vertical, and diagonal straight lines created by the
    addition of a new point.  If there are any extraneous points, the
    intermediate point is replaced with the new endpoint in the curve dictionary.

    Arguments:
        pt1 - first point in segment of curve
        pt2 - second point in segment of curve
        unconnected_pts - dictionary (keys are endpoints of contour curves that
            are not yet finished, values are keys of corresponding curves
        curves - dictionary (keys are arbitrary integers, values are lists of
            lists that represents points on contour curves
        dimensions - tuple representing dimensions of image (row, col)

    Returns:
        A boolean representing whether a point was replaced or not. """

    replaced = False
    # Horizontal segment
    if pt1[0] == pt2[0] and curves[curve_id][0][pt_index] == pt2[0]:
        if pt2[1] != int(pt2[1]):
            change_pt(pt2, pt_index, curves, curve_id)
        else:
            change_pt(pt1, pt_index, curves, curve_id)
        replaced = True

    # Vertical segment
    elif pt1[1] == pt2[1] and curves[curve_id][1][pt_index] == pt2[1]:
        if pt2[0] != int(pt2[0]):
            change_pt(pt2, pt_index, curves, curve_id)
        else:
            change_pt(pt1, pt_index, curves, curve_id)
        replaced = True

    # Slanted segment in upper right or lower left corner
    elif pt2[0] == pt1[0] - 0.5 and pt2[1] == pt1[1] - 0.5:
        if (curves[curve_id][1][pt_index] == pt2[1] -
            (pt2[0] - curves[curve_id][0][pt_index])):
            change_pt(pt1, pt_index, curves, curve_id)
            replaced = True

    # Slanted segment in upper left corner.
    # Does not connect 2 curves since this function isn't called by adjoin_curves
    elif (pt1[1] == int(pt1[1]) and pt2[0] == pt1[0] - 0.5 and pt2[1] == pt1[1]
          + 0.5):
        # Determine which pt is connected to the curve given
        pt1_curve = None
        if pt_index == 1:
            pt1_curve = (curves[curve_id][0][0] == pt1[0] and
                         curves[curve_id][1][0] == pt1[1])
        else:
            pt1_curve = (curves[curve_id][0][-1] == pt1[0] and
                         curves[curve_id][1][-1] == pt1[1])

        if pt1_curve:
            if (curves[curve_id][0][pt_index] == pt1[0] + 0.5 and
                curves[curve_id][1][pt_index] == pt1[1] - 0.5):
                change_pt(pt2, pt_index, curves, curve_id)
                replaced = True
        else:
            if (curves[curve_id][1][pt_index] == pt2[1] +
                (pt2[0] - curves[curve_id][0][pt_index])):
                change_pt(pt1, pt_index, curves, curve_id)
                replaced = True
    return replaced


# Removes extraneous points when connecting two curves
def rm_adjoining_pts(pt1, pt2, curve1_id, pt1_index, curve2_id, pt2_index,
                     curves):
    # Checks if previous point on curve 2 renders pt2 extraneous
    if (curves[curve2_id][1][pt2_index] == pt2[1] +
        (pt2[0] - curves[curve2_id][0][pt2_index])):
        rm_pt(curve2_id, pt2_index, curves)

    # Checks if previous point on curve 1 renders pt1 extraneous
    if (pt1[0] == curves[curve1_id][0][pt1_index] - 0.5 and pt1[1] ==
        curves[curve1_id][1][pt1_index] + 0.5):
        rm_pt(curve1_id, pt1_index, curves)


# Removes extraneous point for add_pt
def change_pt(repl_pt, pt_index, curves, curve_id):
    if pt_index == 1:
        curves[curve_id][0][0] = repl_pt[0]
        curves[curve_id][1][0] = repl_pt[1]
    else:
        curves[curve_id][0][-1] = repl_pt[0]
        curves[curve_id][1][-1] = repl_pt[1]


# Removes extraneous point for rm_adjoining_pts
def rm_pt(curve_id, pt_index, curves):
    if pt_index == 1:
        curves[curve_id][0].popleft()
        curves[curve_id][1].popleft()
    else:
        curves[curve_id][0].pop()
        curves[curve_id][1].pop()


# Returns the pts that must be added to contour curves
def check_cell(i, j, image):
    cell_pts = []
    # Check right
    if image[i][j + 1] != image[i + 1][j + 1]:
        cell_pts.append((i + 0.5, j + 1))
    # Check up
    if image[i + 1][j + 1] != image[i + 1][j]:
        cell_pts.append((i + 1, j + 0.5))
    # Check left
    if image[i + 1][j] != image[i][j]:
        cell_pts.append((i + 0.5, j))
    # Check down
    if image[i][j] != image[i][j + 1]:
        cell_pts.append((i, j + 0.5))
    return cell_pts


# Returns the number of labels in the neighborhood of a point
def count_labels(i, j, image):
    count = 1
    labels = [image[i][j]]
    if not image[i][j + 1] in labels:
        labels.append(image[i][j + 1])
        count += 1
    if not image[i + 1][j + 1] in labels:
        labels.append(image[i + 1][j + 1])
        count += 1
    if not image[i + 1][j] in labels:
        labels.append(image[i + 1][j])
        count += 1
    return count


# Returns boolean as to whether a point lies on outside boundaries of image
def is_boundary_pt(pt, dimensions):
    if pt[0] == 0 or pt[0] == dimensions[0] - 1:
        return True
    elif pt[1] == 0 or pt[1] == dimensions[1] - 1:
        return True
    return False


# Returns boolean as to whether a given point is a junction pt
def is_junction_pt(pt):
    return not pt[0] == int(pt[0]) and not pt[1] == int(pt[1])


# Takes curves and dimension and scales the points to the unit square
def scale(curves, dimensions):
    h = 1./(min(dimensions[0],dimensions[1]) - 1)
    for curve in curves.values():
        for i in xrange(len(curve[0])):
            curve[0][i] = curve[0][i] * h
            curve[1][i] = curve[1][i] * h
