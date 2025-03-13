import matplotlib.pyplot as plt

def draw_the_position_and_points(grid_size=5,path="../fig/grid.png"):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # Define colors for boundary (white) and inner nodes (gray)
    boundary_color = "white"
    inner_color = "gray"

    # Draw circles for nodes
    for i in range(grid_size):
        for j in range(grid_size):
            # Define color based on position (boundary vs. inner)
            if i == 0 or j == 0 or i == grid_size - 1 or j == grid_size - 1:
                color = boundary_color  # Boundary points
                edgecolor = "black"
            else:
                color = inner_color  # Inner points
                edgecolor = "black"
            
            # Draw circle
            circle = plt.Circle((j, grid_size - 1 - i), 0.4, fc=color, ec=edgecolor, lw=1.5)
            ax.add_patch(circle)
            
            # Label the nodes
            ax.text(j, grid_size - 1 - i, f"$v_{{{i}{j}}}$", ha='center', va='center', fontsize=12)

    # Show the grid
    ax.set_title("5x5 Discretization Grid for the Wave Equation", fontsize=14, pad=20)
    plt.grid(False)
    plt.savefig(path)
    plt.show()

if __name__ == "__main__":
    draw_the_position_and_points(grid_size=5)