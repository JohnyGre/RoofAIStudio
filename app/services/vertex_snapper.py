import numpy as np
from typing import List, Dict, Any

class VertexSnapper:
    """
    Snaps close vertices from different polygons together to form a coherent mesh.
    """

    def snap_planes(self, planes: List[Dict[str, Any]], threshold_px: float) -> List[Dict[str, Any]]:
        """
        Takes a list of plane dictionaries, snaps their contour vertices, and returns
        the updated list of planes.

        Args:
            planes: List of plane dicts, each with a 'contour' key.
            threshold_px: The distance in pixels to consider vertices "close".

        Returns:
            An updated list of plane dicts with snapped vertices.
        """
        if not planes:
            return []

        # 1. Collect all vertices with references
        all_vertices = []
        for plane_idx, plane in enumerate(planes):
            contour = np.array(plane['contour']).reshape(-1, 2)
            for vertex_idx, vertex in enumerate(contour):
                all_vertices.append({
                    "point": tuple(vertex),
                    "plane_idx": plane_idx,
                    "vertex_idx": vertex_idx
                })

        # 2. Cluster close vertices using a breadth-first search approach
        clusters = []
        visited = [False] * len(all_vertices)
        for i in range(len(all_vertices)):
            if visited[i]:
                continue

            new_cluster = []
            q = [i]
            visited[i] = True
            
            head = 0
            while head < len(q):
                current_idx = q[head]
                head += 1
                new_cluster.append(all_vertices[current_idx])
                p1 = np.array(all_vertices[current_idx]['point'])

                for j in range(current_idx + 1, len(all_vertices)):
                    if visited[j]:
                        continue
                    
                    p2 = np.array(all_vertices[j]['point'])
                    if np.linalg.norm(p1 - p2) < threshold_px:
                        visited[j] = True
                        q.append(j)
            
            if len(new_cluster) > 1:
                clusters.append(new_cluster)

        # 3. Compute new positions for each cluster
        vertex_update_map = {}  # (plane_idx, vertex_idx) -> new_point
        for cluster in clusters:
            points = np.array([v['point'] for v in cluster])
            centroid = np.mean(points, axis=0)
            new_point = tuple(np.round(centroid).astype(int))
            
            for v_info in cluster:
                key = (v_info['plane_idx'], v_info['vertex_idx'])
                vertex_update_map[key] = new_point

        # 4. Update polygons
        updated_planes = []
        for plane_idx, plane in enumerate(planes):
            new_contour = []
            original_contour = np.array(plane['contour']).reshape(-1, 2)
            for vertex_idx, vertex in enumerate(original_contour):
                key = (plane_idx, vertex_idx)
                if key in vertex_update_map:
                    new_contour.append(list(vertex_update_map[key]))
                else:
                    new_contour.append(list(vertex))
            
            updated_plane = plane.copy()
            # Avoid creating duplicate start/end points if snapping merges them
            if len(new_contour) > 1 and new_contour[0] == new_contour[-1]:
                new_contour.pop()

            updated_plane['contour'] = np.array(new_contour).reshape(-1, 1, 2).tolist()
            updated_planes.append(updated_plane)
            
        return updated_planes