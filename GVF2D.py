import numpy as np
import scipy.ndimage as sn


def GVF(f, mu, ITER, tol=2e-2):
    """
    %GVF Compute gradient vector flow.
    %   [u,v] = GVF(f, mu, ITER) computes the
    %   GVF of an edge map f.  mu is the GVF regularization coefficient
    %   and ITER is the number of iterations that will be computed.

    %   Chenyang Xu and Jerry L. Prince 6/17/97
    %   Copyright (c) 1996-99 by Chenyang Xu and Jerry L. Prince
    %   Image Analysis and Communications Lab, Johns Hopkins University

    %   modified on 9/9/99 by Chenyang Xu
    %   MATLAB do not deal their boundary condition for gradient and del2
    %   consistently between MATLAB 4.2 and MATLAB 5. Hence I modify
    %   the function to take care of this issue by the code itself.
    %   Also, in the previous version, the input "f" is assumed to have been
    %   normalized to the range [0,1] before the function is called.
    %   In this version, "f" is normalized inside the function to avoid
    %   potential error of inputing an unnormalized "f".
    """

    fmin = np.min(f[:, :])
    fmax = np.max(f[:, :])
    f = (f - fmin) / (fmax - fmin)  # % Normalize f to the range [0,1]

    f = BoundMirrorExpand(f)  # % Take care of boundary condition
    [fx, fy] = np.gradient(f)  # % Calculate the gradient of the edge map
    u = fx
    v = fy  # % Initialize GVF to the gradient
    SqrMagf = fx * fx + fy * fy  # % Squared magnitude of the gradient field

    converged = False
    # % Iteratively solve for the GVF u,v
    for i in range(ITER):
        u_prev = u.copy()
        v_prev = v.copy()

        u = BoundMirrorEnsure(u)
        v = BoundMirrorEnsure(v)
        u = u + mu * sn.laplace(u) - SqrMagf * (u - fx)
        v = v + mu * sn.laplace(v) - SqrMagf * (v - fy)
        # Convergence check
        delta = np.linalg.norm(u - u_prev) + np.linalg.norm(v - v_prev)
        print(delta)
        if delta < tol:
            converged = True

            break

    u = BoundMirrorEnsure(u)
    v = BoundMirrorEnsure(v)
    u = BoundMirrorShrink(u)
    v = BoundMirrorShrink(v)

    return u, v, converged


def BoundMirrorEnsure(A):
    """
    % Ensure mirror boundary condition          %
    % The number of rows and columns of A must be greater than 2
    %
    % for example (X means value that is not of interest)
    %
    % A = [
    %     X  X  X  X  X   X
    %     X  1  2  3  11  X
    %     X  4  5  6  12  X
    %     X  7  8  9  13  X
    %     X  X  X  X  X   X
    %     ]
    %
    % B = BoundMirrorEnsure(A) will yield
    %
    %     5  4  5  6  12  6
    %     2  1  2  3  11  3
    %     5  4  5  6  12  6
    %     8  7  8  9  13  9
    %     5  4  5  6  12  6
    %

    % Chenyang Xu and Jerry L. Prince, 9/9/1999
    % http://iacl.ece.jhu.edu/projects/gvf
    """
    m, n = A.shape

    if m < 3 or n < 3:
        raise ValueError("Either the number of rows or columns is smaller than 3.")

    # Copy A to B to avoid modifying the original matrix
    B = A.copy()

    # Mirror the top and bottom rows
    B[0, :] = B[2, :]
    B[-1, :] = B[-3, :]

    # Mirror the left and right columns
    B[:, 0] = B[:, 2]
    B[:, -1] = B[:, -3]

    # Mirror the corners
    # Top-left corner
    B[0, 0] = B[2, 2]
    # Top-right corner
    B[0, -1] = B[2, -3]
    # Bottom-left corner
    B[-1, 0] = B[-3, 2]
    # Bottom-right corner
    B[-1, -1] = B[-3, -3]

    return B


def BoundMirrorExpand(A):
    """
    % Expand the matrix using mirror boundary condition
    %
    % for example
    %
    % A = [
    %     1  2  3  11
    %     4  5  6  12
    %     7  8  9  13
    %     ]
    %
    % B = BoundMirrorExpand(A) will yield
    %
    %     5  4  5  6  12  6
    %     2  1  2  3  11  3
    %     5  4  5  6  12  6
    %     8  7  8  9  13  9
    %     5  4  5  6  12  6
    %

    % Chenyang Xu and Jerry L. Prince, 9/9/1999
    % http://iacl.ece.jhu.edu/projects/gvf
    """

    # shift for matlab style

    m, n = A.shape

    # Create an expanded matrix with additional rows and columns
    B = np.zeros((m + 2, n + 2))

    # Place the original matrix in the center of the expanded matrix
    B[1:-1, 1:-1] = A

    # Mirror the top and bottom rows
    B[0, 1:-1] = B[2, 1:-1]
    B[-1, 1:-1] = B[-3, 1:-1]

    # Mirror the left and right columns
    B[1:-1, 0] = B[1:-1, 2]
    B[1:-1, -1] = B[1:-1, -3]

    # Mirror the corners
    # Top-left corner
    B[0, 0] = B[2, 2]
    # Top-right corner
    B[0, -1] = B[2, -3]
    # Bottom-left corner
    B[-1, 0] = B[-3, 2]
    # Bottom-right corner
    B[-1, -1] = B[-3, -3]

    return B


def BoundMirrorShrink(A):
    """
    % Shrink the matrix to remove the padded mirror boundaries
    %
    % for example
    %
    % A = [
    %     5  4  5  6  12  6
    %     2  1  2  3  11  3
    %     5  4  5  6  12  6
    %     8  7  8  9  13  9
    %     5  4  5  6  12  6
    %     ]
    %
    % B = BoundMirrorShrink(A) will yield
    %
    %     1  2  3  11
    %     4  5  6  12
    %     7  8  9  13

    % Chenyang Xu and Jerry L. Prince, 9/9/1999
    % http://iacl.ece.jhu.edu/projects/gvf
    """

    [m, n] = A.shape
    yi = np.arange(1, m - 1)
    xi = np.arange(1, n - 1)
    B = A[np.ix_(yi, xi)]

    return B
