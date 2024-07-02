import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Code by Mary (Evelyn) van den Akker, and Kyle Smith
# Last edited 6/6/2024

def run():
    # Material properties, pick any non-negative properties
    med1 = 5
    med2 = 2
    med3 = 3

    # This for loop exists, so I could generate a bunch of frames to
    # animate the eigenvalues, right now it only runs one loop
    # it isn't needed, but I will leave it anyway
    for z in range(1, 2):

        # Forward Wave
        def Jf(a, b):
            return (a + b) / (2 * a)

        # Backwards Wave
        def Jb(a, b):
            return (b - a) / (2 * a)

        # Transmitted Wave
        def Jt(a, b):
            return (2 * a) / (b + a)

        # Reflected Wave
        def Jr(a, b):
            return (b - a) / (b + a)


        # Number of cells
        c = 300
        # Number of points
        e = 4
        # Number of total points
        p = e * c
        # Green function
        # Indexed from 0-3
        # d loop
        M = np.zeros((p, p))


        # The form this indexing takes is the following:
        # $$
        # M[((NUMBER OF POINTS * (d - WHICH CELL THE WAVE ENDS IN)) + WHICH POINT IT ENDS AT) % p, NUMBER OF POINTS * d]
        # $$
        # You only have to use the mod (%) operator if the wave ends in a different cell from the one it started in!
        # These coincide with my field pattern. look at the "3 Medium Field Pattern" picture in the GitHub for reference
        for d in range(0, c):
            # Current injected at point 0
            # 2,3,4 of previous cell
            M[((4 * (d - 1)) + 1) % p, 4 * d] += Jb(med1, med2) * Jt(med2, med1) * Jt(med1, med3) * Jf(med3, med1)
            M[((4 * (d - 1)) + 2) % p, 4 * d] += Jb(med1, med2) * Jt(med2, med1) * Jt(med1, med3) * Jb(med3, med1)
            M[((4 * (d - 1)) + 3) % p, 4 * d] += Jb(med1, med2) * Jr(med2, med1) * Jb(med2, med3) * Jt(med3, med1) + Jf(med1, med2) * Jr(med2, med1) * Jf(med2, med3) * Jt(med3, med1)

            # 1,2,3,4 of current cell
            M[((4 * (d + 0)) + 0), 4 * d] += Jb(med1, med2) * Jt(med2, med1) * Jr(med1, med3)
            M[((4 * (d + 0)) + 1), 4 * d] += Jb(med1, med2) * Jr(med2, med1) * (Jb(med2, med3) * Jr(med3, med1) * Jb(med3, med1) + Jf(med2, med3) * Jr(med3, med1) * Jf(med3,med1)) + Jf(med1, med2) * Jr(med2, med1) * (Jf(med2, med3) * Jr(med3, med1) * Jf(med3, med1) + Jb(med2, med3) * Jr(med3, med1) * Jb(med3, med1))
            M[((4 * (d + 0)) + 2), 4 * d] += Jb(med1, med2) * Jr(med2, med1) * (Jb(med2, med3) * Jr(med3, med1) * Jf(med3, med1) + Jf(med2, med3) * Jr(med3, med1) * Jb(med3,med1)) + Jf(med1, med2) * Jr(med2, med1) * (Jf(med2, med3) * Jr(med3, med1) * Jb(med3, med1) + Jb(med2, med3) * Jr(med3, med1) * Jf(med3, med1))
            M[((4 * (d + 0)) + 3), 4 * d] += Jf(med1, med2) * Jt(med2, med1) * Jr(med1, med3)

            # 1,2,3 of next cell
            M[((4 * (d + 1)) + 0) % p, 4 * d] += Jb(med1, med2) * Jr(med2, med1) * Jf(med2, med3) * Jt(med3, med1) + Jf(med1, med2) * Jr(med2, med1) * Jb(med2, med3) * Jt(med3, med1)
            M[((4 * (d + 1)) + 1) % p, 4 * d] += Jf(med1, med2) * Jt(med2, med1) * Jt(med1, med3) * Jb(med3, med1)
            M[((4 * (d + 1)) + 2) % p, 4 * d] += Jf(med1, med2) * Jt(med2, med1) * Jt(med1, med3) * Jf(med3, med1)

            # Current injected at point 1
            # 4 of previous, previous cell
            M[((4 * (d - 2)) + 3) % p, 4 * d + 1] += Jt(med1, med2) * Jf(med2, med3) * Jt(med3, med1)

            # 2,3,4 of previous cell
            M[((4 * (d - 1)) + 1) % p, 4 * d + 1] += Jt(med1, med2) * (Jf(med2, med3) * Jr(med3, med1) * Jb(med3, med1) + Jb(med2, med3) * Jr(med3, med1) * Jf(med3,med1))
            M[((4 * (d - 1)) + 2) % p, 4 * d + 1] += Jt(med1, med2) * (Jf(med2, med3) * Jr(med3, med1) * Jf(med3, med1) + Jb(med2, med3) * Jr(med3, med1) * Jb(med3,med1))
            M[((4 * (d - 1)) + 3) % p, 4 * d + 1] += Jr(med1, med2) * Jr(med1, med3)

            # 1,2,3 of current cell
            M[(4 * (d + 0) + 0), 4 * d + 1] += Jt(med1, med2) * Jb(med2, med3) * Jt(med3, med1)
            M[(4 * (d + 0) + 1), 4 * d + 1] += Jr(med1, med2) * Jt(med3, med3) * Jb(med3, med1)
            M[(4 * (d + 0) + 2), 4 * d + 1] += Jr(med1, med2) * Jt(med2, med3) * Jf(med3, med1)

            # Current injected at point 2
            # 2,3,4 of current cell
            M[(4 * (d + 0) + 1), 4 * d + 2] += Jr(med1, med2) * Jr(med1, med3)
            M[(4 * (d + 0) + 2), 4 * d + 2] += Jt(med1, med2) * (Jb(med2, med3) * Jr(med3, med1) * Jb(med3, med1) + Jf(med2, med3) * Jr(med3, med1) * Jf(med3,med1))
            M[(4 * (d + 0) + 3), 4 * d + 2] += Jt(med1, med2) * (Jb(med2, med3) * Jr(med3, med1) * Jf(med3, med1) + Jf(med2, med3) * Jr(med3, med1) * Jb(med3,med1))

            # 1,2,3 of next cell
            M[((4 * (d + 1)) + 0) % p, 4 * d + 2] += Jb(med1, med2) * Jr(med2, med1) * Jf(med2, med3) * Jt(med3,med1) + Jf(med1, med2) * Jr(med2, med1) * Jb(med2, med3) * Jt(med3, med1)
            M[((4 * (d + 1)) + 1) % p, 4 * d + 2] += Jf(med1, med2) * Jt(med2, med1) * Jt(med1, med3) * Jb(med3, med1)
            M[((4 * (d + 1)) + 2) % p, 4 * d + 2] += Jf(med1, med2) * Jt(med2, med1) * Jt(med1, med3) * Jf(med3, med1)

            # 1 of next, next cell
            M[((4 * (d + 2)) + 0) % p, 4 * d + 2] += Jt(med1, med2) * Jf(med2, med3) * Jt(med3, med1)

            # Current injected at point 3
            # 2,3,4 of previous cell
            M[((4 * (d - 1)) + 1) % p, 4 * d + 3] += Jf(med1, med2) * Jt(med2, med1) * Jt(med1, med3) * Jf(med3, med1)
            M[((4 * (d - 1)) + 2) % p, 4 * d + 3] += Jf(med1, med2) * Jt(med2, med1) * Jt(med1, med3) * Jb(med3, med1)
            M[((4 * (d - 1)) + 3) % p, 4 * d + 3] += Jf(med1, med2) * Jr(med2, med1) * Jb(med2, med3) * Jt(med3,med1) + Jf(med1, med2) * Jr(med2, med1) * Jf(med2, med3) * Jt(med3, med1)

            # 1,2,3,4 of current cell
            M[((4 * (d + 0)) + 0), 4 * d + 3] += Jf(med1, med2) * Jt(med2, med1) * Jr(med1, med3)
            M[((4 * (d + 0)) + 1), 4 * d + 3] += Jf(med1, med2) * Jr(med2, med1) * (Jb(med2, med3) * Jr(med3, med1) * Jb(med3, med1) + Jf(med2, med3) * Jr(med3, med1) * Jf(med3,med1)) + Jb(med1, med2) * Jr(med2, med1) * (Jf(med2, med3) * Jr(med3, med1) * Jb(med3, med1) + Jb(med2, med3) * Jr(med3, med1) * Jf(med3, med1))
            M[((4 * (d + 0)) + 2), 4 * d + 3] += Jf(med1, med2) * Jr(med2, med1) * (Jb(med2, med3) * Jr(med3, med1) * Jf(med3, med1) + Jf(med2, med3) * Jr(med3, med1) * Jb(med3,med1)) + Jb(med1, med2) * Jr(med2, med1) * (Jf(med2, med3) * Jr(med3, med1) * Jf(med3, med1) + Jb(med2, med3) * Jr(med3, med1) * Jb(med3, med1))
            M[((4 * (d + 0)) + 3), 4 * d + 3] += Jb(med1, med2) * Jt(med2, med1) * Jr(med1, med3)

            # 1,2,3 of next cell
            M[((4 * (d + 1)) + 0) % p, 4 * d + 3] += (Jf(med1, med2) * Jr(med2, med1) * Jf(med2, med3) * Jt(med3, med1)+ Jb(med1, med2) * Jr(med2, med1) * Jb(med2, med3) * Jt(med3,med1))
            M[((4 * (d + 1)) + 1) % p, 4 * d + 3] += Jb(med1, med2) * Jt(med2, med1) * Jt(med1, med3) * Jb(med3, med1)
            M[((4 * (d + 1)) + 2) % p, 4 * d + 3] += Jb(med1, med2) * Jt(med2, med1) * Jt(med1, med3) * Jf(med3, med1)

        # Calculating the Eigenvalues and Eigenvectors
        eigValues, eigVectors = np.linalg.eig(M)

        # Extract the eigenvalues
        z = eigValues

        # Calculate the magnitude of the eigenvalues
        modul = np.abs(z)

        # Initialize arrays to store the real and imaginary parts
        rrr = np.real(z)
        iii = np.imag(z)

        # Prepare the unit circle for plotting
        ang = np.arange(0, 2 * np.pi, 0.01)
        xc = np.cos(ang)
        yc = np.sin(ang)

        # Plot the unit circle and the eigenvalues
        plt.figure()
        plt.plot(xc, yc, label='Unit Circle')  # Plot the unit circle
        plt.plot(rrr, iii, '.', markersize=3, label='Eigenvalues')  # Plot the eigenvalues
        plt.xlabel(r'Real$(\lambda)$', fontsize=14)
        plt.ylabel(r'Imag$(\lambda)$', fontsize=14)

        # Plot the titles
        plt.title("Eigenvalues for n = 300 case with med1 = {0}, med2 = {1}, med3 = {2}".format(med1, med2, med3))
        plt.axis('equal')
        plt.legend()

        t = 20  # Number of time steps

        # Initialize matrices
        m = np.zeros((t, p))

        # Set up the injected Current

        #v = eigVectors[p//2] # Use this if you want to plot an eigenvector
        #v /= np.linalg.norm(v)

        # -Creates a vector of the whole length with the midpoint set to 1
        v = np.zeros(p)
        v[p // 2 + 1] = 1

        # "Time" (n) evolution
        # This is where the propigation is calculated, google the transfer matrix method
        for n in range(t):
            v = np.power(M, n + 1).dot(v)
            # v /= np.linalg.norm(v)  # Normalize at each step
            m[n, :] = np.real(v)

        # Define space and time axes
        x = np.linspace(1, p, p)
        y = np.linspace(1, t, t)

        # Create a 3D mesh plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, m, cmap='viridis')

        # Labels
        ax.set_xlabel('space', fontsize=14)
        ax.set_ylabel('n', fontsize=14)
        ax.set_zlabel('amplitude of the wave', fontsize=14)

        # Show the plot
        plt.show()

        # This shows you your greens matrix
        plt.imshow(M, cmap='Greens', interpolation='nearest')
        plt.colorbar()  # Optional: adds a colorbar to indicate the value scale
        plt.title("Greens Matrix")
        plt.show()

        # These exist to increment the mediums for the animation frames
        # med1 += 0.05
        # med2 += 0.05
        # med3 += 0.05


if __name__ == '__main__':
    run()

# Hey this is Kyle, if you get too confused send me a text at Three85-8Three1-98Four5. I can help
