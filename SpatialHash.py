import numpy as np
from collections import defaultdict
from typing import Dict, Set, Tuple, List

class SpatialHashGrid:
    def __init__(self, cell_size: float = 1.0):
        """
        Initialize spatial hash grid
        
        Args:
            cell_size: Size of each grid cell. Should be >= maximum collision distance
        """
        self.cell_size = cell_size
        self.vertex_grid: Dict[Tuple[int, int, int], Set[Tuple[int, int]]] = defaultdict(set)  # (body_idx, vertex_idx)
        self.triangle_grid: Dict[Tuple[int, int, int], Set[Tuple[int, int]]] = defaultdict(set)  # (body_idx, triangle_idx)
        
    def hash_position(self, position: np.ndarray) -> Tuple[int, int, int]:
        """Convert 3D position to grid cell indices"""
        return (
            int(np.floor(position[0] / self.cell_size)),
            int(np.floor(position[1] / self.cell_size)),
            int(np.floor(position[2] / self.cell_size))
        )
    
    def get_cells_for_aabb(self, min_pos: np.ndarray, max_pos: np.ndarray) -> List[Tuple[int, int, int]]:
        """Get all grid cells that an AABB intersects"""
        min_cell = self.hash_position(min_pos)
        max_cell = self.hash_position(max_pos)
        
        cells = []
        for x in range(min_cell[0], max_cell[0] + 1):
            for y in range(min_cell[1], max_cell[1] + 1):
                for z in range(min_cell[2], max_cell[2] + 1):
                    cells.append((x, y, z))
        return cells
    
    def get_triangle_aabb(self, vertices: np.ndarray, triangle_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute AABB for a triangle"""
        triangle_vertices = vertices[triangle_indices]
        min_pos = np.min(triangle_vertices, axis=0)
        max_pos = np.max(triangle_vertices, axis=0)
        return min_pos, max_pos
    
    def update_grid(self, bodies: list) -> None:
        """
        Update the spatial hash grid with new body positions
        
        Args:
            bodies: List of soft body objects
        """
        
        # Clear previous grid
        self.vertex_grid.clear()
        self.triangle_grid.clear()
        
        # Insert all vertices and triangles into grid
        for body_idx, body in enumerate(bodies):
            
            # Insert vertices
            vertex_cells = set()
            for vertex_idx, position in enumerate(body.positions):
                cell = self.hash_position(position)
                self.vertex_grid[cell].add((body_idx, vertex_idx))
                vertex_cells.add(cell)
            
            # Insert triangles
            triangle_cells = set()
            for triangle_idx, triangle in enumerate(body.mesh.faces):
                min_pos, max_pos = self.get_triangle_aabb(body.positions, triangle)
                cells = self.get_cells_for_aabb(min_pos, max_pos)
                for cell in cells:
                    self.triangle_grid[cell].add((body_idx, triangle_idx))
                    triangle_cells.add(cell)
    
    def find_collision_candidates(self, exclude_body_idx: int = None) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Find potential vertex-triangle collision pairs for the specified body
        
        Args:
            exclude_body_idx: The body whose vertices we're checking
            
        Returns:
            List of ((body_idx, vertex_idx), (body_idx, triangle_idx)) pairs
        """
        candidates = []
        
        # We're only interested in vertices of the current body
        for cell in self.vertex_grid:
            vertices = {(b_idx, v_idx) for b_idx, v_idx in self.vertex_grid[cell] 
                       if b_idx == exclude_body_idx}
            
            if not vertices:
                continue
                
            # Get triangles from this cell and neighboring cells
            triangles = set()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        neighbor = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                        # Only get triangles from other bodies
                        triangles.update({(b_idx, t_idx) for b_idx, t_idx in self.triangle_grid.get(neighbor, set())
                                        if b_idx != exclude_body_idx})
            
            # Create vertex-triangle pairs
            for body_idx, vertex_idx in vertices:
                for other_body_idx, triangle_idx in triangles:
                    candidates.append(((body_idx, vertex_idx), (other_body_idx, triangle_idx)))
        
        return candidates
    
    def compute_optimal_cell_size(self, bodies: list) -> float:
        """
        Compute optimal cell size based on mesh statistics
        """
        total_edge_length = 0
        edge_count = 0
        min_edge = float('inf')
        max_edge = 0
        
        # Collect edge statistics
        for body in bodies:
            for edge in body.edgeIds:
                p1 = body.positions[edge[0]]
                p2 = body.positions[edge[1]]
                length = np.linalg.norm(p2 - p1)
                total_edge_length += length
                edge_count += 1
                min_edge = min(min_edge, length)
                max_edge = max(max_edge, length)
        
        avg_edge = total_edge_length / edge_count if edge_count > 0 else 1.0
        
        # Compute mesh bounds for scale reference
        all_positions = np.vstack([body.positions for body in bodies])
        bounds_min = np.min(all_positions, axis=0)
        bounds_max = np.max(all_positions, axis=0)
        diagonal = np.linalg.norm(bounds_max - bounds_min)
        
        # Print debug info
        print(f"\nMesh Statistics:")
        print(f"Average edge length: {avg_edge:.4f}")
        print(f"Min edge length: {min_edge:.4f}")
        print(f"Max edge length: {max_edge:.4f}")
        print(f"Mesh diagonal: {diagonal:.4f}")
        
        # Try different cell sizes and estimate costs
        multipliers = [0.5, 1.0, 2.0, 4.0]
        best_multiplier = 1.0
        best_cost = float('inf')
        
        for m in multipliers:
            cell_size = avg_edge * m
            estimated_cells = int(np.ceil(diagonal / cell_size)) ** 3
            avg_objects_per_cell = (edge_count * 2) / estimated_cells  # rough estimate
            
            # Cost model: balance between number of cells and objects per cell
            grid_cost = estimated_cells  # Memory and iteration cost
            query_cost = avg_objects_per_cell ** 2  # Collision check cost
            total_cost = grid_cost + query_cost
            
            print(f"\nMultiplier {m}:")
            print(f"Cell size: {cell_size:.4f}")
            print(f"Estimated cells: {estimated_cells}")
            print(f"Avg objects per cell: {avg_objects_per_cell:.2f}")
            print(f"Estimated cost: {total_cost:.2f}")
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_multiplier = m
        
        optimal_size = avg_edge * best_multiplier
        print(f"\nChosen optimal cell size: {optimal_size:.4f}")
        return optimal_size

    def analyze_grid_performance(self):
        """
        Analyze current grid statistics
        """
        print("\nGrid Performance Analysis:")
        print(f"Cell size: {self.cell_size}")
        print(f"Number of occupied cells: {len(self.vertex_grid)}")
        
        # Vertex distribution
        vertex_counts = [len(vertices) for vertices in self.vertex_grid.values()]
        if vertex_counts:
            print(f"Vertices per cell - Min: {min(vertex_counts)}, "
                  f"Max: {max(vertex_counts)}, "
                  f"Avg: {sum(vertex_counts)/len(vertex_counts):.2f}")
        
        # Triangle distribution
        triangle_counts = [len(triangles) for triangles in self.triangle_grid.values()]
        if triangle_counts:
            print(f"Triangles per cell - Min: {min(triangle_counts)}, "
                  f"Max: {max(triangle_counts)}, "
                  f"Avg: {sum(triangle_counts)/len(triangle_counts):.2f}")