import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob as glob_module
import typer

from pathlib import Path
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

cli = typer.Typer(add_completion=False)


def plot_3d(df: pd.DataFrame, filename: Path, arrow_scale: float = 0.05):
    particles_good = df[df['class'] != -9999]
    particles_bad  = df[df['class'] == -9999]
    particles_flipped = df[df['class'] == -1]

    # normalize positions between [0,1]
    positions_all = df[['tx','ty','tz']].to_numpy()
    positions_min = positions_all.min(axis=0)
    positions_max = positions_all.max(axis=0)
    positions_range = positions_max - positions_min

    positions_good = (particles_good[['tx', 'ty', 'tz']].to_numpy() - positions_min) / positions_range
    positions_bad = (particles_bad[['tx', 'ty', 'tz']].to_numpy() - positions_min) / positions_range
    positions_flipped = (particles_flipped[['tx', 'ty', 'tz']].to_numpy() - positions_min) / positions_range

    # extract orientation vectors
    def extract_orientations(df: pd.DataFrame, flip_z: bool = False) -> NDArray[np.float64]:
        R = df[
            ['r00', 'r01', 'r02',
             'r10', 'r11', 'r12',
             'r20', 'r21', 'r22']
        ].to_numpy(dtype=np.float64).reshape(-1, 3, 3)
        z = R[:, :, 2]
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        if flip_z:
            z *= -1
        return z

    orientations_good = extract_orientations(particles_good)
    orientations_flipped = extract_orientations(particles_flipped, True)

    # color good particles by their lattice id
    unique_lattices = np.sort(np.abs(particles_good['lattice'].unique()))
    n_colors = len(unique_lattices)
    cmap = plt.colormaps['tab20'].resampled(n_colors)
    class_to_cmap_index = {int(lid): i for i, lid in enumerate(unique_lattices)}
    cmap_indices = particles_good['lattice'].map(class_to_cmap_index).to_numpy(dtype=int)

    # plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        positions_good[:, 0], positions_good[:, 1], positions_good[:, 2],
        c=cmap_indices, cmap=cmap, vmin=0, vmax=n_colors - 1,
        s=30, depthshade=True, alpha=1.0, label='good (colored by lattice)'
    )
    ax.scatter(
        positions_bad[:, 0], positions_bad[:, 1], positions_bad[:, 2],
        c='k', s=5, depthshade=True, alpha=0.5, label='bad', zorder=10
    )
    ax.quiver(
        positions_good[:, 0], positions_good[:, 1], positions_good[:, 2],
        orientations_good[:, 0], orientations_good[:, 1], orientations_good[:, 2],
        length=arrow_scale, color='k', linewidth=0.5, arrow_length_ratio=0.1
    )
    ax.quiver(
        positions_flipped[:, 0], positions_flipped[:, 1], positions_flipped[:, 2],
        orientations_flipped[:, 0], orientations_flipped[:, 1], orientations_flipped[:, 2],
        length=arrow_scale, color='r', linewidth=0.5, arrow_length_ratio=0.1
    )

    # Labels & view
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    ax.set_title(f'{filename.stem}')
    ax.set_box_aspect([1, 1, 1])  # equal aspect ratio

    plt.tight_layout()
    plt.show()


def flip_particles(
    df: pd.DataFrame,
    positions: NDArray[np.float64],
    rotation_matrices: NDArray[np.float64],
    orientations: NDArray[np.float64],
    particles_lattice_id: NDArray[np.int_],
    particles_in_lattice: NDArray[np.bool_],
    n_lattices: int,
    z_shift_if_flipped: float,
    plot: bool,
):
    n_particles = len(positions)

    # compute sum of orientations per lattice (all lattices at once)
    lattice_sums = np.zeros((n_lattices, 3))
    for dim in range(3):
        lattice_sums[:, dim] = np.bincount(
            particles_lattice_id, weights=orientations[:, dim], minlength=n_lattices
        )

    # normalize per lattice
    norms = np.linalg.norm(lattice_sums, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    avg_orientations_per_lattice = lattice_sums / norms

    # assign average orientation to each particle
    avg_orientations = avg_orientations_per_lattice[particles_lattice_id]

    # find flipped particles
    dots = np.einsum('ij,ij->i', orientations, avg_orientations)
    flipped_mask = (dots < 0) & particles_in_lattice
    flipped_indices = np.where(flipped_mask)[0]

    if flipped_indices.size > 0:
        print(f'Flipping {flipped_indices.size} particles')
        if plot:
            df.loc[flipped_indices, 'class'] = -1

        # rotate 180 around x
        # TODO when doing so, we mess up the in-place alignment,
        #      maybe adding a rotz180 after the rotx180 would be good
        rotation_matrices[flipped_indices, :, 2] *= -1
        rotation_matrices[flipped_indices, :, 0] *= -1

        # shift positions along original z
        positions[flipped_indices] -= orientations[flipped_indices] * z_shift_if_flipped

    # update rotation matrices in the DataFrame
    df[
        ['r00', 'r01', 'r02',
         'r10', 'r11', 'r12',
         'r20', 'r21', 'r22']
    ] = rotation_matrices.reshape(n_particles, 9)
    df[['tx', 'ty', 'tz']] = positions


def clean_particles(
    df: pd.DataFrame,
    min_distance: float,
    max_distance: float,
    min_orientation: float,
    max_orientation: float,
    min_curvature: float,
    max_curvature: float,
    min_neighbours: int,
    min_lattice_size: int,
    allow_flipped_particles: bool,
    orientation_flip: bool,
    z_shift_if_flipped: float,
    plot: bool,
) -> pd.DataFrame:
    # extract the data
    positions = df[['tx', 'ty', 'tz']].to_numpy(dtype=np.float64)
    rotation_matrices = df[
        ['r00', 'r01', 'r02',
         'r10', 'r11', 'r12',
         'r20', 'r21', 'r22']
    ].to_numpy(dtype=np.float64).reshape(-1, 3, 3)  # (n,3,3)
    n_particles = len(positions)

    # get the orientations from the rotation matrices by taking the z column vectors
    orientations = rotation_matrices[:, :, 2]  # (n,3)
    orientations /= np.linalg.norm(orientations, axis=1, keepdims=True)

    # create a mask for the orientations within the specified range
    orientation_matrix = np.clip(orientations @ orientations.T, -1.0, 1.0)
    if allow_flipped_particles:
        orientation_matrix = np.abs(orientation_matrix)
    orientation_mask = ((orientation_matrix >= min_orientation) &
                        (orientation_matrix <= max_orientation))

    # create a mask for the distances within the specified range
    distance_matrix = positions[:, None, :] - positions[None, :, :]  # (n,n,3)
    distance_matrix_sqd = np.sum(distance_matrix ** 2, axis=-1)  # (n,n)
    distance_mask = ((distance_matrix_sqd >= min_distance ** 2) &
                     (distance_matrix_sqd <= max_distance ** 2))
    np.fill_diagonal(distance_mask, False)  # remove self-match

    # create a mask for the curvature within the specified range
    # the curvature is the dot product of the distance vector and the orientation vector of the neighbor
    norms = np.linalg.norm(distance_matrix, axis=-1, keepdims=True)
    distance_matrix_normalized = np.divide(distance_matrix, norms, out=np.zeros_like(distance_matrix), where=norms > 0)
    curvature_matrix = np.einsum('ijk,jk->ij', distance_matrix_normalized, orientations)  # dot product
    curvature_mask = ((curvature_matrix >= min_curvature) &
                      (curvature_matrix <= max_curvature))

    # combined masks
    neighbor_mask = np.logical_and.reduce([distance_mask, orientation_mask, curvature_mask])  # (n,n)

    # remove particles with too few neighbors
    n_neighbors = neighbor_mask.sum(axis=1)  # (n,)
    valid = n_neighbors >= min_neighbours  # (n,)
    neighbor_mask &= valid[:, None]
    neighbor_mask &= valid[None, :]

    # compute the graph connectivity, i.e., connected particles are assigned the same lattice
    adjacency = csr_matrix(neighbor_mask)
    n_lattices, particles_lattice_id = connected_components(csgraph=adjacency, directed=False)

    # remove small lattices
    n_particles_per_lattice = np.bincount(particles_lattice_id)
    large_lattice_ids = n_particles_per_lattice >= max(1, min_lattice_size)
    particles_in_lattice = large_lattice_ids[particles_lattice_id] & valid  # (n,)
    df.loc[~particles_in_lattice, 'class'] = -9999
    print(f'Particles left={particles_in_lattice.sum()}/{n_particles} '
          f'({n_particles - particles_in_lattice.sum()} removed)')

    # use an empty column to store the lattice id for plotting
    if plot:
        df['lattice'] = particles_lattice_id

    if orientation_flip:
        flip_particles(
            df,
            positions=positions,
            rotation_matrices=rotation_matrices,
            orientations=orientations,
            particles_lattice_id=particles_lattice_id,
            particles_in_lattice=particles_in_lattice,
            n_lattices=n_lattices,
            z_shift_if_flipped=z_shift_if_flipped,
            plot=plot,
        )

    return df


def process_file(
    csv_file: Path,
    min_distance: float,
    max_distance: float,
    angle_tolerance: float,
    curvature_tolerance: float,
    min_neighbours: int,
    min_array_size: int,
    allow_flipped_particles: bool,
    flip_z: bool,
    plot: bool
):
    print(f'Cleaning file: {csv_file}')
    df = pd.read_csv(
        csv_file,
        sep=' ',
        header=None,
        skip_blank_lines=True,
        index_col=False,
        names=[
            'cc', 'sampling', 'e0', 'uid', 'e1', 'e2', 'lattice', 'e4', 'e5', 'e6',
            'tx', 'ty', 'tz', 'angle0', 'angle1', 'angle2',
            'r00', 'r10', 'r20', 'r01', 'r11', 'r21', 'r02', 'r12', 'r22', 'class'
        ]
    )

    # convert for dot product comparison
    angle_tolerance = np.clip(angle_tolerance, 0., 90.)
    min_orientation = np.cos(np.radians(angle_tolerance))
    max_orientation = 1.0

    curvature_tolerance = np.clip(curvature_tolerance, 0., 90.)
    min_curvature = np.sin(np.radians(-curvature_tolerance))
    max_curvature = np.sin(np.radians(+curvature_tolerance))

    df = clean_particles(
        df,
        min_distance,
        max_distance,
        min_orientation,
        max_orientation,
        min_curvature,
        max_curvature,
        min_neighbours,
        min_array_size,
        allow_flipped_particles=allow_flipped_particles,
        orientation_flip=flip_z,
        z_shift_if_flipped=0,
        plot=plot,
    )

    if plot:
        plot_3d(df, filename=csv_file, arrow_scale=0.05)
        df['lattice'] = 1  # TODO we could leave it, emClarity ignores it anyway

    # remove cleaned-out particles from the csv
    df.drop(df[df['class'] == -9999].index, inplace=True)

    # save the cleaned dataframe
    output_file = csv_file.with_stem(csv_file.stem + '_cleaned')
    df.to_csv(output_file, sep=' ', header=False, index=False)
    print(f'Saved cleaned dataframe: {output_file}\n')


@cli.command(no_args_is_help=True, help='Remove particles from emClarity CSV files based on simple geometry constrains')
def main(
    csv_pattern: str = typer.Option(..., help='Path or glob pattern for CSV file(s), e.g., "tilt1_1_bin6.csv" or "convmap/*.csv"'),
    min_distance: float = typer.Option(20, help='Min distance between particles, in unbinned pixels'),
    max_distance: float = typer.Option(120, help='Max distance between particles, in unbinned pixels'),
    angle_tolerance: float = typer.Option(40, help='Angle tolerance between particles, in degrees. Clamped between 0 to 90'),
    allow_flipped_particles: bool = typer.Option(True, help='When filtering based on the angle tolerance, ignore whether the particles are up or down.'),
    curvature_tolerance: float = typer.Option(40, help='Curvature tolerance between particles, in degrees. Clamped between 0 to 90'),
    min_neighbours: int = typer.Option(3, help='Remove particles with less than this number of neighbours'),
    min_array_size: int = typer.Option(6, help='Remove lattices with less than this number of valid particles'),
    flip_z: bool = typer.Option(False, help='Rotate particles 180° around their x-axis if facing opposite to lattice average orientation'),
    plot: bool = typer.Option(False, help='Plot before saving the cleaned file'),
):
    # Get the csv files, handling the shell expansion
    pattern_path = Path(csv_pattern)
    if pattern_path.exists() and pattern_path.is_file():
        csv_files = [str(pattern_path)]
    else:
        csv_files = sorted(glob_module.glob(csv_pattern))

    if not csv_files:
        print(f'Error: No files found matching pattern: {csv_pattern}')
        raise typer.Exit(code=1)

    # Process each file
    for csv_file_str in csv_files:
        process_file(
            csv_file=Path(csv_file_str),
            min_distance=min_distance,
            max_distance=max_distance,
            angle_tolerance=angle_tolerance,
            curvature_tolerance=curvature_tolerance,
            min_neighbours=min_neighbours,
            min_array_size=min_array_size,
            allow_flipped_particles=allow_flipped_particles,
            flip_z=flip_z,
            plot=plot,
        )


if __name__ == '__main__':
    cli()
