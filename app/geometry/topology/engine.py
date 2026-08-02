"""
Topological engine for deterministic roof geometric analysis, edge classification,
and topological validation.
"""
import uuid
from collections import defaultdict
from typing import List, TYPE_CHECKING

from app.geometry.edge import Edge
from app.geometry.topology.models import RoofTopology, EdgeNode, PlaneNode, EdgeType, ValidationIssue, PlaneAdjacency

if TYPE_CHECKING:
    from app.geometry.plane import RoofPlane
    from app.geometry.roof_geometry import RoofGeometry


class RoofTopologyEngine:
    """
    Deterministic topology analyzer for RoofGeometry.
    Constructs connectivity graphs, classifies roof edges, and validates geometry.
    """

    def build(self, geometry: 'RoofGeometry') -> RoofTopology:
        """
        Analyzes the given RoofGeometry and returns its RoofTopology representation.
        """
        if not geometry or not geometry.planes:
            return RoofTopology(planes=[], edges=[], adjacency=[], outer_boundary=[], validation=[])

        classified_edges = self.classify_edges(geometry.planes)
        plane_nodes = self.build_graph(geometry.planes, classified_edges)

        # Find outer boundary edges (eaves)
        outer_boundary_ids = [
            edge.edge_id for edge in classified_edges if edge.edge_type == EdgeType.EAVE
        ]

        # Build adjacency list
        adjacency_map = defaultdict(set)
        for edge in classified_edges:
            if edge.left_plane_id and edge.right_plane_id:
                # Canonical key for the pair of planes
                plane_pair = tuple(sorted((edge.left_plane_id, edge.right_plane_id)))
                adjacency_map[plane_pair].add(edge.edge_id)
        
        adjacency_list = [
            PlaneAdjacency(
                plane_a_id=pair[0],
                plane_b_id=pair[1],
                shared_edge_ids=edge_ids
            ) for pair, edge_ids in adjacency_map.items()
        ]

        # Create the topology object
        topology = RoofTopology(
            planes=plane_nodes,
            edges=classified_edges,
            adjacency=adjacency_list,
            outer_boundary=outer_boundary_ids,
            validation=[]
        )

        # Run validation
        validation_issues = self.validate(geometry, topology)

        # Return a new topology object with validation results
        return RoofTopology(
            planes=topology.planes,
            edges=topology.edges,
            adjacency=topology.adjacency,
            outer_boundary=topology.outer_boundary,
            validation=validation_issues
        )

    def classify_edges(self, planes: List['RoofPlane']) -> List[EdgeNode]:
        """
        Determines edge classification based on how many planes an edge belongs to.
        - 1 plane: EAVE
        - 2 planes: UNKNOWN (to be classified later as RIDGE, VALLEY, or HIP)
        """
        edge_to_planes = defaultdict(list)

        for plane in planes:
            polygon_vertices = plane.polygon.vertices
            for i in range(len(polygon_vertices)):
                p1 = polygon_vertices[i]
                p2 = polygon_vertices[(i + 1) % len(polygon_vertices)]

                # Create a canonical representation for the edge
                if (p1.x, p1.y) > (p2.x, p2.y):
                    p1, p2 = p2, p1
                
                edge = Edge(start_point=p1, end_point=p2)
                edge_to_planes[edge].append(plane.name)

        classified_edges = []
        for edge, plane_names in edge_to_planes.items():
            num_planes = len(plane_names)
            edge_type = EdgeType.UNKNOWN
            left_plane_id = None
            right_plane_id = None

            if num_planes == 1:
                edge_type = EdgeType.EAVE
                left_plane_id = plane_names[0]
            elif num_planes == 2:
                edge_type = EdgeType.UNKNOWN
                left_plane_id = plane_names[0]
                right_plane_id = plane_names[1]

            start_tuple = (edge.start_point.x, edge.start_point.y)
            end_tuple = (edge.end_point.x, edge.end_point.y)
            
            classified_edges.append(
                EdgeNode(
                    edge_id=str(uuid.uuid4()),
                    start=start_tuple,
                    end=end_tuple,
                    edge_type=edge_type,
                    left_plane_id=left_plane_id,
                    right_plane_id=right_plane_id,
                    length=edge.length,
                )
            )
        return classified_edges

    def build_graph(self, planes: List['RoofPlane'], edges: List[EdgeNode]) -> List[PlaneNode]:
        """
        Builds the connectivity graph of planes and edges.
        """
        if not planes:
            return []

        # Initialize nodes for each plane
        plane_nodes = {
            plane.name: {
                "edge_ids": set(),
                "neighbor_plane_ids": set()
            } for plane in planes
        }

        # Populate edges and neighbors from the classified edge list
        for edge in edges:
            if edge.left_plane_id and edge.left_plane_id in plane_nodes:
                plane_nodes[edge.left_plane_id]["edge_ids"].add(edge.edge_id)

            if edge.right_plane_id and edge.right_plane_id in plane_nodes:
                plane_nodes[edge.right_plane_id]["edge_ids"].add(edge.edge_id)

            # If an edge connects two planes, they are neighbors
            if edge.left_plane_id and edge.right_plane_id:
                if edge.left_plane_id in plane_nodes and edge.right_plane_id in plane_nodes:
                    plane_nodes[edge.left_plane_id]["neighbor_plane_ids"].add(edge.right_plane_id)
                    plane_nodes[edge.right_plane_id]["neighbor_plane_ids"].add(edge.left_plane_id)

        # Create the final list of PlaneNode objects
        result = [
            PlaneNode(
                plane_id=plane_name,
                edge_ids=data["edge_ids"],
                neighbor_plane_ids=data["neighbor_plane_ids"]
            )
            for plane_name, data in plane_nodes.items()
        ]

        return result

    def validate(self, geometry: 'RoofGeometry', topology: RoofTopology) -> List[ValidationIssue]:
        """
        Runs a validation suite on the geometric and topological validity of the roof.
        Checks for:
        - Invalid polygons (self-intersection)
        - Orphan planes
        """
        issues = []
        
        # 1. Check for invalid polygons
        if geometry.planes:
            for plane in geometry.planes:
                if not plane.polygon.is_valid:
                    issue = ValidationIssue(
                        severity='error',
                        code='invalid_polygon',
                        message=f"Plane '{plane.name}' has an invalid polygon (e.g., self-intersecting).",
                        offending_ids=[plane.name]
                    )
                    issues.append(issue)

        # 2. Check for orphan planes in multi-plane geometries
        if geometry.planes and len(geometry.planes) > 1:
            plane_names = {plane.name for plane in geometry.planes}
            connected_planes = set()

            for edge_node in topology.edges:
                # An edge is shared if it has two planes connected to it.
                if edge_node.left_plane_id and edge_node.right_plane_id:
                    connected_planes.add(edge_node.left_plane_id)
                    connected_planes.add(edge_node.right_plane_id)
            
            orphan_planes = plane_names - connected_planes
            for plane_name in orphan_planes:
                issues.append(ValidationIssue(
                    severity='error',
                    code='orphan_plane',
                    message=f"Plane '{plane_name}' is not connected to any other plane.",
                    offending_ids=[plane_name]
                ))

        return issues
