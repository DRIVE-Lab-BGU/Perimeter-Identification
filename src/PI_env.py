import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.spatial import ConvexHull, QhullError
import cv2
from NetworkHeatmap import NetworkHeatmap

class SimplifiedJunctionEnv(gym.Env):
    def __init__(self, net, prepared_junctions, heatmap_without_network, heatmap_with_network, lambda_reg, congestion_threshold):
        super().__init__()

        self.net = net
        self.prepared_junctions = prepared_junctions
        self.heatmap_with_network = heatmap_with_network
        self.lambda_reg = lambda_reg
        self.congestion_threshold = congestion_threshold

        # Pre-compute grayscale normalized heatmap once
        if heatmap_without_network.ndim >= 3:
            gray = cv2.cvtColor(heatmap_without_network, cv2.COLOR_BGR2GRAY)
        else:
            gray = heatmap_without_network
        self.norm_heatmap = gray.astype(np.float32) / 255.0

        self.junction_dict = {j['id']: j for j in prepared_junctions}
        self.node_id_to_index = {j['id']: i for i, j in enumerate(prepared_junctions)}
        self.index_to_node_id = {i: j['id'] for i, j in enumerate(prepared_junctions)}
        self.num_junctions = len(prepared_junctions)

        # Add 1 extra action for "finish episode"
        self.action_space = spaces.Discrete(self.num_junctions + 1)
        self.FINISH_ACTION = self.num_junctions  # Last action is finish

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.num_junctions,),
            dtype=np.uint8
        )

        self.active_junctions = set()
        self.current_value = 0.0
        self.previous_value = 0.0  # Track previous perimeter value
        self.convex_hull = None
        self.hull_points = None
        self.count_congested_pixels = 0

        # total congested pixels across the whole heatmap (not just inside hull)
        self.total_congested_count = int(np.count_nonzero(self.norm_heatmap < self.congestion_threshold))

  
    def is_point_on_hull_boundary(self, point, tolerance=3.0):
        """
        Check if a point is on or near (within <tolerance> pixels) 
        the boundary of the convex hull.
        """
        if self.convex_hull is None:
            return False

        equations = self.convex_hull.equations
        
        min_abs_distance = float('inf')
        
        for eq in equations:
            a, b, offset = eq
            distance = a * point[0] + b * point[1] + offset

            # If the point is outside this edge by more than the tolerance
            if distance > tolerance:
                return False

            # Track the minimum absolute distance to any edge
            if abs(distance) < min_abs_distance:
                min_abs_distance = abs(distance)

        # Check if closest edge was within the tolerance
        return min_abs_distance <= tolerance

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        else:
            super().reset()
        
        # Filter for congested junctions (intensity < self.congestion_threshold means congested, since lower = more traffic)
        congested_junctions = [j for j in self.prepared_junctions if j['intensity'] < self.congestion_threshold]
        
        # If we have congested junctions, select the top 80% most congested
        if len(congested_junctions) >= 3:
            # Sort by intensity (ascending - lower intensity = more congested)
            congested_junctions.sort(key=lambda x: x['intensity'])
            
            # Take top 80% most congested - ALL of them
            num_to_select = max(3, int(len(congested_junctions) * 0.8))
            selected = congested_junctions[:num_to_select]
        else:
            # Fallback: if not enough congested junctions, use original approach
            num_to_select = max(3, int(self.num_junctions * 0.03))
            selected = self.prepared_junctions[:num_to_select]
        
        self.active_junctions = set(j['id'] for j in selected)
        self.update_convex_hull()
        
          
        # Activate all junctions inside the convex hull
        self.activate_interior_junctions()
        
        
        h, w = self.norm_heatmap.shape  # Now using pre-computed grayscale
        area = h * w

        mask = np.zeros((h, w), dtype=np.uint8)
        hull_vertices = self.hull_points[self.convex_hull.vertices].reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [hull_vertices], 1)

        inside = mask > 0
        intensities = self.norm_heatmap[inside]  # Direct access, no conversion needed

        congested = intensities < self.congestion_threshold
        self.count_congested_pixels = np.count_nonzero(congested)
        
        
        self.current_value = self.calculate_perimeter_value()
        self.previous_value = self.current_value  # Initialize previous value
        
        return self.get_observation(), {}

    def activate_interior_junctions(self):
        """Ensure all and only junctions inside the convex hull are active (VECTORIZED)."""
        if self.convex_hull is None:
            return
    
        # Get boundary junction coordinates
        hull_vertices = set(tuple(p) for p in self.hull_points[self.convex_hull.vertices])
    
        # Find which active junctions are on the boundary
        boundary_junctions = set()
        for j_id in self.active_junctions:
            pt = (self.junction_dict[j_id]['pixel_x'], self.junction_dict[j_id]['pixel_y'])
            if pt in hull_vertices:
                boundary_junctions.add(j_id)
    
        # Rebuild active set: boundary + interior junctions
        new_active = boundary_junctions.copy()
        
        # VECTORIZED: Check all non-boundary junctions at once
        non_boundary_ids = [j_id for j_id in self.junction_dict.keys() 
                            if j_id not in boundary_junctions]
        
        if non_boundary_ids:
            # Create array of all points to check
            points = np.array([[self.junction_dict[j_id]['pixel_x'], 
                               self.junction_dict[j_id]['pixel_y']] 
                              for j_id in non_boundary_ids])
            
            # Check all at once
            inside_mask = self.are_points_in_hull(points)
            
            # Add interior junctions to active set
            for j_id, is_inside in zip(non_boundary_ids, inside_mask):
                if is_inside:
                    new_active.add(j_id)
    
        self.active_junctions = new_active


    def step(self, action):
        """
        Execute action in the environment.
        Action is the junction index to toggle (activate/deactivate) or finish episode.
        """
        if action == self.FINISH_ACTION:
            # For finish action, reward is 0 (no change in perimeter)
            return self.get_observation(), 0.0, True, False, {"finished_by_action": True}

        j_id = self.index_to_node_id[action]

        if j_id in self.active_junctions:
            # Remove junction if it's active and on the boundary
            if j_id in self.get_valid_remove_candidates() and len(self.active_junctions) > 3:
                # Simply remove the junction - no replacement
                self.active_junctions.remove(j_id)
                self.update_convex_hull()
                # After updating hull, activate any junctions now inside
                self.activate_interior_junctions()
        else:
            # Add junction if it's inactive and outside the hull
            if j_id in self.get_addable_junctions():
                self.active_junctions.add(j_id)
                self.update_convex_hull()
                # After updating hull, activate any junctions now inside
                self.activate_interior_junctions()

        # Calculate new perimeter value
        self.current_value = self.calculate_perimeter_value()

        # Reward is the improvement (current - previous)
        reward = self.current_value - self.previous_value

        # Update previous value for next step
        self.previous_value = self.current_value

        return self.get_observation(), reward, False, False, {}

    def update_convex_hull(self):
        if len(self.active_junctions) < 3:
            self.convex_hull = None
            self.hull_points = None
            return

        points = np.array(
            [[self.junction_dict[j]['pixel_x'], self.junction_dict[j]['pixel_y']] for j in self.active_junctions])
        self.hull_points = points
        
        try:
          self.convex_hull = ConvexHull(points)
        except (QhullError, ValueError) as e:
          self.convex_hull = ConvexHull(points, qhull_options='QJ')

    def calculate_perimeter_value(self):
        if self.convex_hull is None:
            return 0.0

        h, w = self.norm_heatmap.shape  # Now using pre-computed grayscale
        area = h * w

        mask = np.zeros((h, w), dtype=np.uint8)
        hull_vertices = self.hull_points[self.convex_hull.vertices].reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [hull_vertices], 1)

        inside = mask > 0
        intensities = self.norm_heatmap[inside]  # Direct access, no conversion needed

        congested = intensities < self.congestion_threshold
        penalty = np.sum(intensities[~congested])
        value = np.sum(1 -intensities[congested])

        if (self.total_congested_count - self.count_congested_pixels) > 0:
            return 100 * (value - self.lambda_reg * penalty) / ( self.total_congested_count - self.count_congested_pixels)
        else:
            return 100 * (value - self.lambda_reg * penalty)

    def get_observation(self):
        """
        Return a binary observation vector over all junctions.
        Active junctions are marked with 1, inactive with 0.
        
        Returns:
        --------
        np.ndarray of shape (num_junctions,), dtype=np.uint8
        """
        obs = np.zeros(self.num_junctions, dtype=np.uint8)
        for j_id in self.active_junctions:
            idx = self.node_id_to_index[j_id]
            obs[idx] = 1  # Mark junction as active
        return obs

    def _create_binary_grayscale_base(self):
        """
        Create a binary-style grayscale base image:
        - Congested pixels (< congestion_threshold): Dark gray with transparency
        - Non-congested pixels (>= congestion_threshold): White
        """
        h, w = self.norm_heatmap.shape
        
        # Create white background (BGR format)
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Define dark gray color (BGR format)
        dark_gray = np.array([70, 70, 70], dtype=np.uint8)
        
        # Create mask for congested areas
        congested_mask = self.norm_heatmap < self.congestion_threshold
        
        # Apply dark gray to congested areas with transparency
        alpha = 0.6  # Transparency factor
        img[congested_mask] = (alpha * dark_gray + (1 - alpha) * img[congested_mask]).astype(np.uint8)
        
        return img

    def render(self, mode='human'):
        if self.convex_hull is None or self.hull_points is None:
            return

        # Choose base image based on mode
        if mode == 'binary_grayscale':
            # Create binary grayscale image
            img = self._create_binary_grayscale_base()
        else:
            # Use colored heatmap
            img = self.heatmap_with_network.copy()

        # Get boundary junctions only for rendering
        boundary_junctions = self.get_valid_remove_candidates()

        # Only draw boundary junctions
        for j_id in boundary_junctions:
            j = self.junction_dict[j_id]
            cv2.circle(img, (j['pixel_x'], j['pixel_y']), 10, (0, 0, 255), -1)

        for simplex in self.convex_hull.simplices:
            pt1 = tuple(self.hull_points[simplex[0]])
            pt2 = tuple(self.hull_points[simplex[1]])
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)

        if mode == 'human':
            cv2.imshow("Perimeter", img)
            cv2.waitKey(1)
        else:
            return img
            
                
    def _is_dead_end(self, junction_id):
        """
        Check if a junction is a dead-end (has only one connection).
        """
        if junction_id not in self.junction_dict:
            return True
        
        node = self.net.getNode(junction_id)
        
        # Count unique neighbors
        neighbors = set()
        for edge in list(node.getOutgoing()) + list(node.getIncoming()):
            neighbor_id = edge.getToNode().getID() if edge.getFromNode().getID() == junction_id else edge.getFromNode().getID()
            if neighbor_id in self.junction_dict:
                neighbors.add(neighbor_id)
        
        # Dead-end if only 1 neighbor (or 0)
        return len(neighbors) <= 1


    def is_junction_important(self, junction_id):
        """
        Final filtering: Keep only junctions that control access to the congested area.
        
        A junction is important if EITHER:
        1. Junction is OUTSIDE the hull AND has a road going to a junction INSIDE the hull
           (and the neighbor inside is NOT a dead-end)
        OR
        2. Junction is ON the boundary AND has a road going to the interior
           (and the neighbor inside is NOT a dead-end)
        
        In both cases, the junction itself must NOT be a dead-end.
        
        Result: Only junctions ON or OUTSIDE the boundary will be marked red.
        """
        if junction_id not in self.junction_dict:
            return False
        
        j = self.junction_dict[junction_id]
        junction_pt = np.array([j['pixel_x'], j['pixel_y']])
        
        junction_inside = self.is_point_in_hull(junction_pt)
        junction_on_boundary = self.is_point_on_hull_boundary(junction_pt)
        
        # Junction must be ON boundary or OUTSIDE (not inside)
        if junction_inside and not junction_on_boundary:
            return False
        
        # Junction itself must NOT be a dead-end
        if self._is_dead_end(junction_id):
            return False
        
        # Check all connected roads
        node = self.net.getNode(junction_id)
        
        for edge in list(node.getOutgoing()) + list(node.getIncoming()):
            neighbor_id = edge.getToNode().getID() if edge.getFromNode().getID() == junction_id else edge.getFromNode().getID()
            
            if neighbor_id not in self.junction_dict:
                continue
            
            neighbor = self.junction_dict[neighbor_id]
            neighbor_pt = np.array([neighbor['pixel_x'], neighbor['pixel_y']])
            
            neighbor_inside = self.is_point_in_hull(neighbor_pt)
            neighbor_on_boundary = self.is_point_on_hull_boundary(neighbor_pt)
            
            # Check if neighbor is clearly inside (not just on boundary)
            neighbor_clearly_inside = neighbor_inside and not neighbor_on_boundary
            
            # CONDITION 1: Junction is OUTSIDE the hull AND has road to interior
            # AND neighbor inside is NOT a dead-end
            if not junction_inside and not junction_on_boundary and neighbor_clearly_inside:
                # Only count this junction as important if neighbor is NOT a dead-end
                if not self._is_dead_end(neighbor_id):
                    return True
            
            # CONDITION 2: Junction is ON boundary AND road goes to interior
            # AND neighbor inside is NOT a dead-end
            if junction_on_boundary and neighbor_clearly_inside:
                # Only count this junction as important if neighbor is NOT a dead-end
                if not self._is_dead_end(neighbor_id):
                    return True
        
        return False
        
    def render_final_perimeter(self, mode='rgb_array'):
            """
            Renders the perimeter using a Topological Flood Fill (BFS).
            Filters out 'Stranded' Red Nodes that are not connected to any orange road.
            """
            if self.convex_hull is None or self.hull_points is None:
                return
    
            img = self.heatmap_with_network.copy()
    
            # --- 1. Identify Perimeter (Red) Nodes ---
            perimeter_nodes = set()
            for j_id in self.junction_dict.keys():
                if self.is_junction_important(j_id):
                    perimeter_nodes.add(j_id)
    
            if not perimeter_nodes:
                return img if mode != 'human' else None
    
            # --- 2. Find the Seed (Centroid of Convex Hull) ---
            hull_coords = self.hull_points[self.convex_hull.vertices]
            centroid_x = np.mean(hull_coords[:, 0])
            centroid_y = np.mean(hull_coords[:, 1])
            
            start_node_id = None
            min_dist = float('inf')
            
            for j_id, j_data in self.junction_dict.items():
                if j_id in perimeter_nodes:
                    continue
                dist = (j_data['pixel_x'] - centroid_x)**2 + (j_data['pixel_y'] - centroid_y)**2
                if dist < min_dist:
                    min_dist = dist
                    start_node_id = j_id
    
            if start_node_id is None:
                 start_node_id = list(perimeter_nodes)[0]
    
            # --- 3. Run BFS (Flood Fill) ---
            internal_nodes = set()
            queue = [start_node_id]
            visited = {start_node_id}
            
            while queue:
                current_id = queue.pop(0)
                if current_id in perimeter_nodes:
                    continue
                
                internal_nodes.add(current_id)
    
                if self.net.hasNode(current_id):
                    node = self.net.getNode(current_id)
                    edges = list(node.getOutgoing()) + list(node.getIncoming())
                    
                    for edge in edges:
                        neighbor = edge.getToNode() if edge.getFromNode().getID() == current_id else edge.getFromNode()
                        neighbor_id = neighbor.getID()
                        if neighbor_id in self.junction_dict and neighbor_id not in visited:
                            visited.add(neighbor_id)
                            queue.append(neighbor_id)
    
            # --- 4. Draw Orange Roads and Track Connections ---
            bbox = self.net.getBBoxXY()
            
            # New Set: Keep track of which perimeter nodes actually get an cyan road
            connected_perimeter_nodes = set()

            # Track every junction that is an endpoint of a drawn (cyan) road
            cyan_road_junctions = set()
    
            for edge in self.net.getEdges():
                from_id = edge.getFromNode().getID()
                to_id = edge.getToNode().getID()
    
                is_start_in = from_id in internal_nodes
                is_end_in = to_id in internal_nodes
                is_start_perim = from_id in perimeter_nodes
                is_end_perim = to_id in perimeter_nodes
    
                # Original Logic: Only draw if connected to an Internal Node
                should_draw = (is_start_in and (is_end_in or is_end_perim)) or \
                              (is_end_in and (is_start_in or is_start_perim))
    
                if should_draw:
                    # 1. Register connections
                    if is_start_perim:
                        connected_perimeter_nodes.add(from_id)
                    if is_end_perim:
                        connected_perimeter_nodes.add(to_id)

                    # Record both endpoints as being connected to a cyan road
                    cyan_road_junctions.add(from_id)
                    cyan_road_junctions.add(to_id)
                    
                    # 2. Draw the road
                    shape = edge.getShape()
                    if shape:
                        pixel_points = [NetworkHeatmap.coord_to_pixel(x, y, bbox, img.shape) for x, y in shape]
                        pts = np.array(pixel_points, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img, [pts], False, (255, 255, 0), 2)

            # --- 4b. Rescue bridging convex-hull vertices (single pass) ---
            # A convex-hull vertex that ended up NOT red, is NOT touching any cyan
            # road, but sits between >= 2 red junctions, is promoted to red. This
            # closes gaps in the perimeter ring.

            # Identify the convex-hull vertex junctions (pre-post-processing perimeter)
            hull_coords = set(tuple(p) for p in self.hull_points[self.convex_hull.vertices])
            hull_vertex_nodes = set()
            for j_id in self.active_junctions:
                jd = self.junction_dict[j_id]
                if (jd['pixel_x'], jd['pixel_y']) in hull_coords:
                    hull_vertex_nodes.add(j_id)

            # Snapshot of the red set so this stays a single pass (a junction
            # rescued here cannot help rescue another in the same pass).
            red_before = set(connected_perimeter_nodes)
            newly_red = set()

            for j_id in hull_vertex_nodes:
                # Skip if already red or if it touches a cyan road
                if j_id in red_before or j_id in cyan_road_junctions:
                    continue
                if not self.net.hasNode(j_id):
                    continue

                # Collect road-connected neighbors and count how many are red
                node = self.net.getNode(j_id)
                neighbors = set()
                for edge in list(node.getOutgoing()) + list(node.getIncoming()):
                    nb = edge.getToNode().getID() if edge.getFromNode().getID() == j_id \
                        else edge.getFromNode().getID()
                    if nb in self.junction_dict:
                        neighbors.add(nb)

                if len(neighbors & red_before) >= 2:
                    newly_red.add(j_id)

            connected_perimeter_nodes |= newly_red

            # --- 4c. Close the ring: draw CYAN roads between any two red junctions ---
            for edge in self.net.getEdges():
                from_id = edge.getFromNode().getID()
                to_id = edge.getToNode().getID()

                if from_id in connected_perimeter_nodes and to_id in connected_perimeter_nodes:
                    shape = edge.getShape()
                    if shape:
                        pixel_points = [NetworkHeatmap.coord_to_pixel(x, y, bbox, img.shape) for x, y in shape]
                        pts = np.array(pixel_points, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img, [pts], False, (255, 255, 0), 2)

            # --- 5. Draw Red Perimeter Nodes (Filtered) ---
            # We now iterate over 'connected_perimeter_nodes' instead of all 'perimeter_nodes'
            for j_id in connected_perimeter_nodes:
                j = self.junction_dict[j_id]
                cv2.circle(img, (j['pixel_x'], j['pixel_y']), 10, (0, 0, 255), -1)
    
            if mode == 'human':
                cv2.imshow("Final Perimeter", img)
                cv2.waitKey(1)
            else:
                return img
    def get_valid_remove_candidates(self):
        if self.convex_hull is None or len(self.active_junctions) <= 3:
            return set()

        hull_coords = set(tuple(p) for p in self.hull_points[self.convex_hull.vertices])
        valid_ids = set()

        for j_id in self.active_junctions:
            pt = (self.junction_dict[j_id]['pixel_x'], self.junction_dict[j_id]['pixel_y'])
            if pt in hull_coords:
                valid_ids.add(j_id)

        return valid_ids

    def get_addable_junctions(self):
        """Return inactive junctions that are 1 step away from ANY active junction.
        Since all interior junctions are automatically active, inactive junctions
        are guaranteed to be outside the convex hull."""
        if self.convex_hull is None:
            return set()

        candidates = set()
        visited = set()

        # Check neighbors of ALL active junctions
        for j_id in self.active_junctions:
            node = self.net.getNode(j_id)

            # Check outgoing neighbors
            for edge in node.getOutgoing():
                neighbor = edge.getToNode().getID()
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor not in self.active_junctions and neighbor in self.junction_dict:
                        # If it's not active, it's outside the hull (by design)
                        candidates.add(neighbor)

            # Check incoming neighbors
            for edge in node.getIncoming():
                neighbor = edge.getFromNode().getID()
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor not in self.active_junctions and neighbor in self.junction_dict:
                        # If it's not active, it's outside the hull (by design)
                        candidates.add(neighbor)

        return candidates

    def is_point_in_hull(self, point):
        """Check if a single point is inside the convex hull (vectorized)."""
        if self.convex_hull is None:
            return False
        
        eq = self.convex_hull.equations
        # eq is [N, 3], point is [2]
        # We want: eq[:, 0]*x + eq[:, 1]*y + eq[:, 2]
        val = np.dot(eq[:, :-1], point) + eq[:, -1]
        
        # If all values <= epsilon, point is inside
        return np.all(val <= 1e-10)
                
    def are_points_in_hull(self, points):
        """Check multiple points at once (vectorized across points).
        
        Parameters:
        -----------
        points : np.ndarray of shape [N, 2]
            Array of (x, y) coordinates
            
        Returns:
        --------
        np.ndarray of shape [N] with boolean values
        """
        if self.convex_hull is None:
            return np.zeros(len(points), dtype=bool)
        
        # points: [N, 2], equations: [M, 3]
        # Result: [N, M] where each row is one point tested against all equations
        eq = self.convex_hull.equations
        vals = points @ eq[:, :-1].T + eq[:, -1]  # [N, M]
        
        return np.all(vals <= 1e-10, axis=1)  # [N] boolean array

    def get_valid_actions(self):
        """
        Return a list of valid action indices at the current state.
        Includes junctions that can be added (outside hull, adjacent to active),
        junctions that can be removed (on the hull boundary), and the finish action.
        
        Returns:
        --------
        list of int
        """
        actions = []

        # Can add junctions that are outside and nearby any active junction
        for j_id in self.get_addable_junctions():
            actions.append(self.node_id_to_index[j_id])

        # Can remove junctions that are on the boundary
        for j_id in self.get_valid_remove_candidates():
            actions.append(self.node_id_to_index[j_id])

        # Always can finish the episode
        actions.append(self.FINISH_ACTION)

        return actions

    def get_action_mask(self):
        """
        Return a binary mask of shape (num_junctions + 1,) where 1 indicates
        a valid action and 0 indicates an invalid action. The last element
        corresponds to the finish action, which is always valid.
        
        Returns:
        --------
        np.ndarray of shape (num_junctions + 1,), dtype=np.uint8
        """
        mask = np.zeros(self.num_junctions + 1, dtype=np.uint8)  # +1 for finish action
        for action in self.get_valid_actions():
            mask[action] = 1
        return mask

    def close(self):
        cv2.destroyAllWindows()
