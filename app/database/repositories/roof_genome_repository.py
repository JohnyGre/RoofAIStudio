"""
This module defines the repository interface and SQLAlchemy implementation
for RoofGenome data access.
"""

import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import NoResultFound

from app.database.models.roof_genome_data import RoofGenome as ORMRoofGenome
from app.geometry.roof_genome import RoofGenome as DomainRoofGenome # Domain model from geometry
import numpy as np

class RoofGenomeRepository(ABC):
    """
    Abstract base class defining the interface for RoofGenome data access operations.
    """

    @abstractmethod
    def add_genome(self, genome: DomainRoofGenome) -> DomainRoofGenome:
        """Adds a new roof genome to the repository."""
        pass

    @abstractmethod
    def get_genome(self, genome_id: uuid.UUID) -> Optional[DomainRoofGenome]:
        """Retrieves a roof genome by its ID."""
        pass

    @abstractmethod
    def update_genome(self, genome: DomainRoofGenome) -> DomainRoofGenome:
        """Updates an existing roof genome in the repository."""
        pass

    @abstractmethod
    def delete_genome(self, genome_id: uuid.UUID) -> None:
        """Deletes a roof genome by its ID."""
        pass

    @abstractmethod
    def find_similar_genomes(self, query_genome: DomainRoofGenome, limit: int = 10) -> List[DomainRoofGenome]:
        """
        Finds similar roof genomes based on their feature vector.
        """
        pass

class SQLAlchemyRoofGenomeRepository(RoofGenomeRepository):
    """
    SQLAlchemy implementation of the RoofGenomeRepository interface.
    """

    def __init__(self, session: Session):
        self.session = session

    def _orm_to_domain_genome(self, orm_genome: ORMRoofGenome) -> DomainRoofGenome:
        """Converts an ORM RoofGenome to a domain RoofGenome."""
        feature_vector_np: Optional[np.ndarray] = None
        if orm_genome.feature_vector:
            try:
                feature_vector_np = np.frombuffer(orm_genome.feature_vector, dtype=np.float32)
            except Exception as e:
                print(f"Warning: Could not deserialize feature vector for genome {orm_genome.id}: {e}")

        return DomainRoofGenome(
            id=orm_genome.id,
            number_of_planes=orm_genome.plane_count,
            number_of_edges=orm_genome.edge_count,
            number_of_ridges=orm_genome.ridge_count,
            number_of_valleys=orm_genome.valley_count,
            number_of_hips=orm_genome.hip_count,
            number_of_openings=orm_genome.opening_count,
            average_slope=orm_genome.average_slope,
            symmetry_score=orm_genome.symmetry_score,
            complexity_score=orm_genome.complexity_score,
            feature_vector=feature_vector_np
        )

    def _domain_to_orm_genome(self, domain_genome: DomainRoofGenome, orm_genome: Optional[ORMRoofGenome] = None) -> ORMRoofGenome:
        """Converts a domain RoofGenome to an ORM RoofGenome."""
        if orm_genome is None:
            orm_genome = ORMRoofGenome(id=domain_genome.id) # Pass ID from domain model

        orm_genome.plane_count = domain_genome.number_of_planes
        orm_genome.edge_count = domain_genome.number_of_edges
        orm_genome.ridge_count = domain_genome.number_of_ridges
        orm_genome.valley_count = domain_genome.number_of_valleys
        orm_genome.hip_count = domain_genome.number_of_hips
        orm_genome.opening_count = domain_genome.number_of_openings
        orm_genome.average_slope = domain_genome.average_slope
        orm_genome.symmetry_score = domain_genome.symmetry_score
        orm_genome.complexity_score = domain_genome.complexity_score

        if domain_genome.feature_vector is not None:
            orm_genome.feature_vector = domain_genome.feature_vector.astype(np.float32).tobytes()
        else:
            orm_genome.feature_vector = None

        return orm_genome

    def add_genome(self, genome: DomainRoofGenome) -> DomainRoofGenome:
        orm_genome = self._domain_to_orm_genome(genome)
        self.session.add(orm_genome)
        self.session.flush() # Ensure ID is set if auto-generated, or object is persisted
        return self._orm_to_domain_genome(orm_genome)

    def get_genome(self, genome_id: uuid.UUID) -> Optional[DomainRoofGenome]:
        orm_genome = self.session.get(ORMRoofGenome, genome_id)
        if orm_genome:
            return self._orm_to_domain_genome(orm_genome)
        return None

    def update_genome(self, genome: DomainRoofGenome) -> DomainRoofGenome:
        try:
            orm_genome = self.session.execute(select(ORMRoofGenome).filter_by(id=genome.id)).scalar_one()
        except NoResultFound:
            raise ValueError(f"RoofGenome with ID {genome.id} not found for update.")
        
        orm_genome = self._domain_to_orm_genome(genome, orm_genome)
        self.session.add(orm_genome)
        self.session.flush()
        return self._orm_to_domain_genome(orm_genome)

    def delete_genome(self, genome_id: uuid.UUID) -> None:
        orm_genome = self.session.get(ORMRoofGenome, genome_id)
        if orm_genome:
            self.session.delete(orm_genome)
            self.session.flush()

    def find_similar_genomes(self, query_genome: DomainRoofGenome, limit: int = 10) -> List[DomainRoofGenome]:
        """
        Finds similar roof genomes based on their feature vector using cosine similarity.
        This is a basic implementation and can be optimized with vector databases for large datasets.
        """
        if query_genome.feature_vector is None:
            print("Query genome has no feature vector. Cannot perform similarity search.")
            return []

        query_vector = query_genome.feature_vector.flatten()
        if np.linalg.norm(query_vector) == 0:
            print("Query genome feature vector is zero. Cannot perform similarity search.")
            return []

        all_orm_genomes = self.session.execute(select(ORMRoofGenome)).scalars().all()
        similarities = []

        for orm_g in all_orm_genomes:
            if orm_g.feature_vector and orm_g.id != query_genome.id: # Exclude self
                target_vector = np.frombuffer(orm_g.feature_vector, dtype=np.float32).flatten()
                if np.linalg.norm(target_vector) > 0:
                    # Cosine similarity
                    similarity = np.dot(query_vector, target_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(target_vector))
                    similarities.append((similarity, orm_g))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [self._orm_to_domain_genome(g) for _, g in similarities[:limit]]