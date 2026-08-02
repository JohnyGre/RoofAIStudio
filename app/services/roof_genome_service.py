"""
This module defines the RoofGenomeService for generating, comparing, and managing roof genomes.
"""

from typing import List, Optional
import numpy as np
import uuid

from app.geometry.roof_geometry import RoofGeometry
from app.geometry.roof_genome import RoofGenome as DomainRoofGenome # Domain model from app.geometry
from app.database.repositories.roof_genome_repository import RoofGenomeRepository
from app.core.logger import setup_logging

logger = setup_logging()

class RoofGenomeService:
    """
    Service for generating a numerical fingerprint (genome) of a roof's geometry,
    comparing genomes for similarity, and managing their persistence.
    """

    def __init__(self, repository: RoofGenomeRepository):
        self._repository = repository
        logger.info("RoofGenomeService initialized.")

    def generate_genome(self, roof_geometry: RoofGeometry) -> DomainRoofGenome:
        """
        Generates a RoofGenome (numerical fingerprint) from a RoofGeometry object.

        Args:
            roof_geometry (RoofGeometry): The geometric model of the roof.

        Returns:
            DomainRoofGenome: The generated roof genome.
        """
        logger.debug("Generating roof genome from RoofGeometry...")

        # Extract numerical features from RoofGeometry
        plane_count = len(roof_geometry.planes)
        edge_count = len(roof_geometry.edges)
        ridge_count = len(roof_geometry.ridges)
        valley_count = len(roof_geometry.valleys)
        # Placeholder for hip_count - needs actual geometric analysis to determine
        hip_count = 0 # For now, assume 0 or derive from edges
        opening_count = len(roof_geometry.openings)

        # Calculate average slope (simplified for now, could be weighted by area)
        total_slope = sum(p.slope for p in roof_geometry.planes)
        average_slope = total_slope / plane_count if plane_count > 0 else 0.0

        # Placeholder for symmetry and complexity scores
        symmetry_score = 0.0 # To be implemented based on geometric analysis (e.g., comparing plane properties)
        complexity_score = float(plane_count + ridge_count + valley_count + hip_count + opening_count) # Simple heuristic

        # Feature vector for AI similarity search (e.g., a flattened array of key features)
        # This vector should contain all numerical features that define the genome.
        feature_vector_data = np.array([
            plane_count, edge_count, ridge_count, valley_count, hip_count, opening_count,
            average_slope, symmetry_score, complexity_score
        ], dtype=np.float32)

        genome = DomainRoofGenome(
            id=uuid.uuid4(), # Generate a new ID for the domain model
            number_of_planes=plane_count,
            number_of_edges=edge_count,
            number_of_ridges=ridge_count,
            number_of_valleys=valley_count,
            number_of_hips=hip_count,
            number_of_openings=opening_count,
            average_slope=average_slope,
            symmetry_score=symmetry_score,
            complexity_score=complexity_score,
            feature_vector=feature_vector_data
        )
        logger.info(f"Generated roof genome: {genome}")
        return genome

    def calculate_similarity(self, genome1: DomainRoofGenome, genome2: DomainRoofGenome) -> float:
        """
        Calculates a similarity score between two roof genomes using cosine similarity of their feature vectors.

        Args:
            genome1 (DomainRoofGenome): The first roof genome.
            genome2 (DomainRoofGenome): The second roof genome.

        Returns:
            float: A similarity score between 0.0 and 1.0.
        """
        if genome1.feature_vector is None or genome2.feature_vector is None:
            logger.warning("Cannot calculate similarity: one or both genomes are missing feature vectors.")
            return 0.0

        vec1 = genome1.feature_vector.flatten()
        vec2 = genome2.feature_vector.flatten()

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0 # Avoid division by zero

        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        # Cosine similarity is between -1 and 1. Normalize to 0-1 range.
        normalized_similarity = (similarity + 1) / 2

        logger.debug(f"Calculated similarity between genomes {genome1.id} and {genome2.id}: {normalized_similarity:.2f}")
        return normalized_similarity

    def save_genome(self, genome: DomainRoofGenome) -> DomainRoofGenome:
        """
        Saves a roof genome to the repository.
        """
        logger.info(f"Saving roof genome with ID: {genome.id}")
        return self._repository.add_genome(genome)

    def find_similar_genomes(self, query_genome: DomainRoofGenome, limit: int = 10) -> List[DomainRoofGenome]:
        """
        Finds similar roof genomes from the repository based on the query genome.
        """
        logger.info(f"Finding similar genomes for query genome: {query_genome.id}")
        return self._repository.find_similar_genomes(query_genome, limit)

    def get_genome_by_id(self, genome_id: uuid.UUID) -> Optional[DomainRoofGenome]:
        """Retrieves a roof genome by its ID."""
        return self._repository.get_genome(genome_id)
