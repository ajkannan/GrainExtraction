"""Finds the grains in an image given the curves and junctions.

Last updated 1/4/12"""

import copy

def find_grains(curves, boundary_curves, junctions):
    """ Finds and returns a dictionary of grains.

    The algorithm to find curves is based on the fact that internal curves
    can be traversed twice, and external curves only once.  A set of external
    curves is established, and one is popped to begin the algorithm.  Each
    grain is formed by either left and straight turns only, or right and straight
    turns only.  As edges are used in grains, they are either taken out of list
    of external curves or made external curves if they can be used once more.
    When the set of external curves is exhausted, the algorithm is done and all
    grains are found.

    Arguments:
        curves - dictionary of curves {curve_id: list of x and y coordinates}
        boundary_curves - a set of curve ids which denote which curves make up
            the boundary of the image
        junctions - a dictionary of junctions and the curves that meet
            {junction point: set of curve ids representing intersecting curves}

    Returns:
        Dictionary of grains {grain id: lists of points that make up grain} """

    active_edges = set(boundary_curves) # Set of edges that can only be used once
    grains = {} # Dictionary of grains {grain id: lists of x and y coordinates}
    grain_id = 0
    while active_edges: # Iterates until there are no more external curves left.
        current_edge_id = active_edges.pop()
        # If there is more than one grain in the image, then there are sets in
        # junctions from which the popped edge must be removed.
        if active_edges:
            rm_from_junctions(current_edge_id, curves, junctions)
        edge = curves[current_edge_id]
        current_grain = copy.deepcopy(edge)
        endpt = -1 # Stores the index of the last point in the current curve
        turn = None # Represents a right or left turning line
        # Loops until one grain is found
        while (edge[0][endpt], edge[1][endpt]) != (current_grain[0][0], current_grain[1][0]):
            next_edges = junctions[(edge[0][endpt], edge[1][endpt])]
            # Find the orientation at the end of the current edge
            current_dir = None
            if endpt == -1:
                current_dir = (find_orientation(edge[0][-2], edge[1][-2],
                                                edge[0][-1], edge[1][-1]))
            else:
                current_dir = (find_orientation(edge[0][1], edge[1][1],
                                                edge[0][0], edge[1][0]))

            best = None # Holds best option for next edge to add to grain
            backup = None # Stores info for straight option if no other is found
            # Examines the two or three possibilities to find the next edge to
            # be added to the grain
            for i in next_edges:
                if current_edge_id != i: # Checks that grain will not double-back
                    # Find which end of the possible edge should be examined
                    # and the orientation of that edge.
                    edge_position = find_edge_info(curves, edge, endpt, i)
                    next_pt = edge_position[0]
                    next_dir = edge_position[1]

                    # Finds angle between current grain & potential next edge
                    # In order of decreasing sharpness
                    # Left turns: 3, 2, 1; Right turns: 5, 6, 7
                    angle = next_dir - current_dir
                    if angle < 0:
                        angle += 8
                    # Updates the best choice for the next edge to choose
                    # Finds the best choice on which way the curve must turn
                    # (left or right) and angle formed by the grain and the curve
                    # possibility.  Stores the best option.
                    if turn == "left" and angle >= 5:
                        best = (i, angle, next_pt)
                    elif turn == "right" and angle >= 1 and angle <= 3:
                        best = (i, angle, next_pt)
                    elif turn == None:
                        if angle >= 1 and angle <= 3:
                            turn = "right"
                        elif angle != 0:
                            turn = "left"
                        best = (i, angle, next_pt)
                    elif turn != None and best == None and angle == 0:
                        backup = (i, 0, next_pt)
            # If there is no other option, the grain will progress forward
            if best == None:
                best = backup

            # Removes duplicate point before adding another edge to current_grain
            current_grain[0].pop()
            current_grain[1].pop()

            # Copies the edge to add into the grain.
            edge_copy = ([copy.copy(curves[best[0]][0]),
                          copy.copy(curves[best[0]][1])])

            # Takes out unnecessary points if straight line
            # NOTE: comment out the function call below to include extra pts
            rm_extra_pts(best, edge_copy)

            # Appends edge onto grain based on indices that intersect
            update_grain(current_grain, edge_copy, best[2])

            # Updates the current edge for the next iteration
            edge = curves[best[0]]
            current_edge_id = best[0]
            # Update endpt to reflect index of the last point for the new edge
            if best[2] == 0:
                endpt = -1
            else:
                endpt = 0

            # Takes out edges that have been traversed twice
            # Adds edges to active edges if used for the first time
            if best[0] in active_edges:
                active_edges.remove(best[0])
                rm_from_junctions(best[0], curves, junctions)
            else:
                active_edges.add(best[0])

        # Removes extra point between starting and ending pieces of grain
        # NOTE: comment out the function call below to include extra pts
        rm_extra_endpts(current_grain)

        # Adds the completed grain to the dictionary of grains
        grains[grain_id] = current_grain
        grain_id += 1
    return grains


# Adds the next edge to the existing part of the grain
def update_grain(current_grain, edge_copy, next_pt):
    # Decides how to add the edge to the grain based on the orientation of the edge
    if next_pt == 0:
        current_grain[0].extend(edge_copy[0])
        current_grain[1].extend(edge_copy[1])
    else:
        current_grain[0].extend(reversed(edge_copy[0]))
        current_grain[1].extend(reversed(edge_copy[1]))


# Finds the orientation of the edge and the endpt that must be examined.
def find_edge_info(curves, edge, endpt, i):
    # Determines whether the new edge to be added must be added backwards
    # or forwards, then determines the angle.  Returns index of next curve
    # and the angle in a tuple.
    if (curves[i][0][0] == edge[0][endpt] and
        curves[i][1][0] == edge[1][endpt]):
        next_dir = (find_orientation(edge[0][endpt], edge[1][endpt],
                                     curves[i][0][1], curves[i][1][1]))
        return (0, next_dir)

    next_dir = (find_orientation(edge[0][endpt], edge[1][endpt],
                                 curves[i][0][-2], curves[i][1][-2]))
    return (-1, next_dir)


# Checks for and removes extra points in the middle of straight lines when
# adding new edges.  Does not check for extraneous points at the
# intersection of the beginning and ending pieces of grains
def rm_extra_pts(best, edge_copy):
    if best[1] == 0:
        if best[2] == -1:
            edge_copy[0].pop()
            edge_copy[1].pop()
        else:
            edge_copy[0].popleft()
            edge_copy[1].popleft()


# Avoids extraneous point if starting and ending in middle of straight line
def rm_extra_endpts(grain):
    # Finds the angle between the first and last piece of the grain
    init_orientation = (find_orientation(grain[0][0], grain[1][0], grain[0][1],
                                         grain[1][1]))
    end_orientation = (find_orientation(grain[0][-2], grain[1][-2],
                                        grain[0][-1], grain[1][-1]))
    final_angle = end_orientation - init_orientation

    # Checks if starting and ending form a straight line
    if final_angle == 0:
        # If so, remove initial pt and adjust end to reflect new starting pt
        grain[0].popleft()
        grain[1].popleft()
        grain[0].pop()
        grain[1].pop()
        grain[0].append(grain[0][0])
        grain[1].append(grain[1][0])


# Returns the orientation between two points (from pt 1 to pt 2)
# Numbers 0-7 represent 8 directions, 0 = North, numbers progressing clockwise
def find_orientation(i1, j1, i2, j2):
    # Horizontal
    if i1 == i2:
        if j2 > j1:
            return 2
        else:
            return 6

    # Vertical
    elif j1 == j2:
        if i2 > i1:
            return 4
        else:
            return 0

    # Upper left/lower right diagonal
    elif j2 == j1 + (i1 - i2):
        if j2 > j1:
            return 1
        else:
            return 5

    # Upper right/lower left diagonal
    elif i2 > i1:
        return 3
    else:
        return 7


# Removes an edge from junctions once it has used the maximum number of times
def rm_from_junctions(edge_id, curves, junctions):
    junctions[(curves[edge_id][0][0], curves[edge_id][1][0])].remove(edge_id)
    junctions[(curves[edge_id][0][-1], curves[edge_id][1][-1])].remove(edge_id)
