# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import (
    Dict,
    Set,
    Iterable,
    Iterator,
    Tuple,
    Any,
    Optional,
)


class UndirectedGraph:
    """
    A minimal yet full-featured container for an *undirected* graph.

    The graph is represented as an adjacency list:
        node -> {neighbor₁, neighbor₂, ...}

    Key points
    ----------
    *  Adding an edge (u, v) automatically adds the reverse edge (v, u).
    *  Nodes can be any hashable Python object.
    *  All operations run in amortised O(1) time except iteration, which is O(V+E).

    Examples
    --------
    >>> g = UndirectedGraph()
    >>> g.add_edge(1, 2)
    >>> g.add_edge(2, 3)
    >>> list(g.nodes())
    [1, 2, 3]
    >>> list(g.neighbors(2))
    [1, 3]
    >>> g.degree(2)
    2
    >>> 2 in g
    True
    >>> g.remove_node(2)
    >>> list(g.edges())
    []
    """

    # -------------------------------------------------------------------------
    # Construction / Deserialization
    # -------------------------------------------------------------------------

    def __init__(self, edges: Optional[Iterable[Tuple[Any, Any]]] = None) -> None:
        """
        Create a new graph.

        Parameters
        ----------
        edges : iterable of (u, v) pairs, optional
            Initial edges to populate the graph.
        """
        self._adj: Dict[Any, Set[Any]] = {}
        if edges is not None:
            for u, v in edges:
                self.add_edge(u, v)

    # -------------------------------------------------------------------------
    # Dunder helpers
    # -------------------------------------------------------------------------

    def __contains__(self, node: Any) -> bool:
        """True if *node* is present."""
        return node in self._adj

    def __iter__(self) -> Iterator[Any]:
        """Iterate over all nodes (same as .nodes())."""
        return iter(self._adj)

    def __len__(self) -> int:
        """Number of nodes."""
        return len(self._adj)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f"{cls}({list(self.edges())})"

    # -------------------------------------------------------------------------
    # Mutators
    # -------------------------------------------------------------------------

    def add_node(self, node: Any) -> None:
        """Add an isolated node (idempotent)."""
        if node not in self._adj:
            self._adj[node] = set()

    def add_edge(self, u: Any, v: Any) -> None:
        """
        Add an undirected edge u-v (idempotent).

        Automatically adds missing endpoints.
        """
        self.add_node(u)
        self.add_node(v)
        self._adj[u].add(v)
        self._adj[v].add(u)

    def remove_node(self, node: Any) -> None:
        """
        Remove *node* and all incident edges.

        Raises
        ------
        KeyError
            If *node* is not present.
        """
        if node not in self._adj:
            raise KeyError(node)

        # Remove *node* from all neighbours
        for neighbour in self._adj[node]:
            self._adj[neighbour].remove(node)
        del self._adj[node]

    def remove_edge(self, u: Any, v: Any) -> None:
        """
        Remove the edge u-v.

        Raises
        ------
        KeyError
            If either endpoint or the edge itself is missing.
        """
        try:
            self._adj[u].remove(v)
            self._adj[v].remove(u)
        except KeyError as exc:
            raise KeyError(f"Edge ({u}, {v}) not found") from exc

    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------

    def nodes(self) -> Iterator[Any]:
        """Iterate over all nodes."""
        return iter(self._adj)

    def edges(self) -> Iterator[Tuple[Any, Any]]:
        """
        Iterate over all edges once each (u ≤ v to avoid duplicates).

        Order is deterministic but arbitrary.
        """
        seen: Set[Tuple[Any, Any]] = set()
        for u in self._adj:
            for v in self._adj[u]:
                if (v, u) not in seen:
                    seen.add((u, v))
                    yield (u, v)

    def neighbors(self, node: Any) -> Iterator[Any]:
        """Iterate over all neighbours of *node*."""
        if node not in self._adj:
            raise KeyError(node)
        return iter(self._adj[node])

    def degree(self, node: Any) -> int:
        """Return the degree (number of neighbours) of *node*."""
        if node not in self._adj:
            raise KeyError(node)
        return len(self._adj[node])

    # -------------------------------------------------------------------------
    # Convenience helpers
    # -------------------------------------------------------------------------

    def copy(self) -> "UndirectedGraph":
        """Return a shallow copy of the graph."""
        g = UndirectedGraph()
        # Deep-copy the sets to keep them independent
        g._adj = {n: set(neighs) for n, neighs in self._adj.items()}
        return g

    def clear(self) -> None:
        """Remove all nodes and edges."""
        self._adj.clear()

    # -------------------------------------------------------------------------
    # Serialization helpers
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[Any, Set[Any]]:
        """
        Return a copy of the underlying adjacency list.

        Useful for JSON serialization after converting sets to lists.
        """
        return {n: set(neighs) for n, neighs in self._adj.items()}

    @classmethod
    def from_dict(cls, data: Dict[Any, Iterable[Any]]) -> "UndirectedGraph":
        """
        Re-create a graph from an adjacency-list-like dictionary.

        Parameters
        ----------
        data : dict
            {node: iterable_of_neighbours}
        """
        g = cls()
        for node, neighbours in data.items():
            for neighbour in neighbours:
                g.add_edge(node, neighbour)
        return g
