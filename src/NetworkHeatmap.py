import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sumolib
from scipy.stats import gaussian_kde
import numpy as np


class NetworkHeatmap:
    @staticmethod
    def prepare_data(net_file, data_file, time_step):
        """
        Load network and prepare KDE data for a specific time step.

        Returns:
        --------
        net : sumolib.net object
        bbox : tuple of (xmin, ymin, xmax, ymax)
        kde_values : 2D array of KDE values
        extent : list of [xmin, xmax, ymin, ymax] for imshow
        """
        # Load SUMO road network and vehicle data
        net = sumolib.net.readNet(net_file)
        df = pd.read_csv(data_file, delimiter=";")

        # Select a specific timestep and compute kernel density estimation (KDE)
        df_filtered = df[df['timestep_time'] == time_step][['vehicle_x', 'vehicle_y']]
        kde = gaussian_kde(np.vstack([df_filtered['vehicle_x'], df_filtered['vehicle_y']]))

        # Get the bounding box
        bbox = net.getBBoxXY()
        (xmin, ymin), (xmax, ymax) = bbox

        # Create a KDE heatmap
        x = np.linspace(xmin, xmax, 1000)
        y = np.linspace(ymin, ymax, 1000)
        x_grid, y_grid = np.meshgrid(x, y)
        grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])

        # Evaluate KDE on this grid
        kde_values = kde(grid_coords).reshape(x_grid.shape)

        extent = [xmin, xmax, ymin, ymax]

        return net, bbox, kde_values, extent

    @staticmethod
    # Function to convert geographical coordinates to image pixel indices
    def coord_to_pixel(x_coord, y_coord, bbox, img_shape):
        x_min, y_min = bbox[0]
        x_max, y_max = bbox[1]

        height, width = img_shape[:2]

        # Scale to [0, 1] range
        x_scaled = (x_coord - x_min) / (x_max - x_min)
        y_scaled = (y_coord - y_min) / (y_max - y_min)

        # Convert to pixel indices
        pixel_x = int(x_scaled * (width - 1))
        # Flip y-axis because image origin is top-left
        pixel_y = int((1 - y_scaled) * (height - 1))

        return pixel_x, pixel_y

    @staticmethod
    def create_heatmap(net, kde_values, extent, show_network=True, output_filename=None):
        """
        Create a heatmap visualization with optional network overlay and thresholding.

        Parameters:
        -----------
        net : sumolib.net object
            The SUMO network object
        kde_values : 2D array
            The KDE values array
        extent : list
            The extent for imshow [xmin, xmax, ymin, ymax]
        show_network : bool, default=True
            Whether to show the road network and junctions
        threshold : float, optional
            If provided, values below this percentile will be displayed, others transparent.
            Thresholded heatmaps are automatically displayed in grayscale.
        output_filename : str, optional
            If provided, the figure will be saved to this filename

        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot road network if requested
        if show_network:
            # Plot road network
            for edge in net.getEdges():
                shape = edge.getShape()
                x_edge, y_edge = zip(*shape)
                ax.plot(x_edge, y_edge, color='gray', linewidth=0.5)

            # Plot junctions
            nodes = net.getNodes()
            for junction in nodes:
                junction_x, junction_y = junction.getCoord()
                ax.scatter(junction_x, junction_y, color="blue", marker="o", s=30)


        # Apply thresholding if requested
        kde_values_display = kde_values.copy()
        # For regular heatmaps, use the Reds colormap
        cmap = sns.color_palette("Reds", as_cmap=True)
        im = ax.imshow(kde_values_display, cmap=cmap, extent=extent, origin='lower', alpha=0.5)

        # Customize plot
        ax.axis('off')

        # Save the figure if requested
        if output_filename:
            plt.savefig(output_filename, bbox_inches='tight', dpi=300, pad_inches=0)

        return fig, ax

    @staticmethod
    def prepare_junctions(net, bbox, heatmap_without_network):
        """
        Convert geographical (x, y) coordinates to image pixel indices.
        
        Parameters:
        -----------
        x_coord : float
            X coordinate in the network's coordinate system
        y_coord : float
            Y coordinate in the network's coordinate system
        bbox : tuple
            Bounding box of the network: ((xmin, ymin), (xmax, ymax))
        img_shape : tuple
            Shape of the target image (height, width, ...)
        
        Returns:
        --------
        pixel_x : int
            Horizontal pixel index
        pixel_y : int
            Vertical pixel index (y-axis is flipped, origin is top-left)
        """
    
        # Ensure the heatmap is grayscale
        if len(heatmap_without_network.shape) >= 3:
            gray_heatmap = cv2.cvtColor(heatmap_without_network, cv2.COLOR_BGR2GRAY)
        else:
            gray_heatmap = heatmap_without_network

        # Collect junction data with their intensity values
        junction_data = []
        for junction in net.getNodes():
            junction_id = junction.getID()
            junction_x, junction_y = junction.getCoord()

            # Convert junction coordinates to image pixel coordinates
            pixel_x, pixel_y = NetworkHeatmap.coord_to_pixel(junction_x, junction_y, bbox, gray_heatmap.shape)

            # Check if pixel coordinates are within bounds
            if (0 <= pixel_y < gray_heatmap.shape[0] and
                    0 <= pixel_x < gray_heatmap.shape[1]):
                # Get pixel intensity (normalized to 0–1)
                intensity = gray_heatmap[pixel_y, pixel_x] / 255.0

                junction_data.append({
                    'id': junction_id,
                    'pixel_x': pixel_x,
                    'pixel_y': pixel_y,
                    'intensity': intensity
                })

        # Sort junctions by intensity (lower intensity = more congested)
        junction_data.sort(key=lambda x: x['intensity'])

        return junction_data