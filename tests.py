"""
Unit-tests for the `UndirectedGraph` container.
"""

import unittest
from undirected_graph import UndirectedGraph


class TestUndirectedGraph(unittest.TestCase):
    """Test cases for every public method of UndirectedGraph."""

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def setUp(self) -> None:
        """Create a fresh fixture before every test."""
        self.g = UndirectedGraph()

    def _populate_triangle(self) -> None:
        """Add the edges of a 3-cycle (triangle)."""
        self.g.add_edge(1, 2)
        self.g.add_edge(2, 3)
        self.g.add_edge(3, 1)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def test_empty_graph(self):
        self.assertEqual(len(self.g), 0)
        self.assertEqual(list(self.g.nodes()), [])
        self.assertEqual(list(self.g.edges()), [])

    def test_from_edges(self):
        g = UndirectedGraph([(1, 2), (2, 3)])
        self.assertEqual(len(g), 3)
        self.assertEqual(set(g.edges()), {(1, 2), (2, 3)})

    def test_from_dict(self):
        g = UndirectedGraph.from_dict({1: [2, 3], 2: [1], 3: [1]})
        self.assertEqual(len(g), 3)
        self.assertEqual(set(g.edges()), {(1, 2), (1, 3)})

    # ------------------------------------------------------------------ #
    # Node existence & containment
    # ------------------------------------------------------------------ #

    def test_contains(self):
        self.g.add_node("A")
        self.assertIn("A", self.g)
        self.assertNotIn("B", self.g)

    def test_len_iter(self):
        self.g.add_edge("X", "Y")
        self.assertEqual(len(self.g), 2)
        self.assertEqual(set(self.g), {"X", "Y"})

    # ------------------------------------------------------------------ #
    # Adding & removing
    # ------------------------------------------------------------------ #

    def test_add_node_idempotent(self):
        self.g.add_node("A")
        self.g.add_node("A")
        self.assertEqual(len(self.g), 1)

    def test_add_edge_creates_nodes(self):
        self.g.add_edge(1, 2)
        self.assertEqual(len(self.g), 2)
        self.assertEqual(self.g.degree(1), 1)
        self.assertEqual(self.g.degree(2), 1)

    def test_add_edge_idempotent(self):
        self.g.add_edge(1, 2)
        self.g.add_edge(1, 2)
        self.assertEqual(self.g.degree(1), 1)
        self.assertEqual(len(self.g), 2)

    def test_remove_node(self):
        self._populate_triangle()
        self.g.remove_node(2)
        self.assertNotIn(2, self.g)
        self.assertEqual(self.g.degree(1), 1)  # only 3 remains
        self.assertEqual(self.g.degree(3), 1)

    def test_remove_node_missing(self):
        with self.assertRaises(KeyError):
            self.g.remove_node("ghost")

    def test_remove_edge(self):
        self.g.add_edge(1, 2)
        self.g.remove_edge(1, 2)
        self.assertEqual(self.g.degree(1), 0)
        self.assertEqual(self.g.degree(2), 0)

    def test_remove_edge_missing(self):
        self.g.add_node(1)
        self.g.add_node(2)
        with self.assertRaises(KeyError):
            self.g.remove_edge(1, 2)

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def test_neighbors(self):
        self._populate_triangle()
        self.assertEqual(set(self.g.neighbors(1)), {2, 3})

    def test_neighbors_missing(self):
        with self.assertRaises(KeyError):
            list(self.g.neighbors("no-such-node"))

    def test_degree(self):
        self._populate_triangle()
        self.assertEqual(self.g.degree(1), 2)

    def test_edges_unique(self):
        self._populate_triangle()
        edges = list(self.g.edges())
        self.assertEqual(len(edges), 3)
        # Ensure no duplicates such as (1,2) and (2,1)
        self.assertEqual(len(set(edges)), 3)

    # ------------------------------------------------------------------ #
    # Copy & clear
    # ------------------------------------------------------------------ #

    def test_copy(self):
        self.g.add_edge("A", "B")
        g2 = self.g.copy()
        g2.remove_node("A")
        self.assertIn("A", self.g)
        self.assertNotIn("A", g2)

    def test_clear(self):
        self._populate_triangle()
        self.g.clear()
        self.assertEqual(len(self.g), 0)

    # ------------------------------------------------------------------ #
    # Serialization round-trip
    # ------------------------------------------------------------------ #

    def test_to_from_dict(self):
        self._populate_triangle()
        data = self.g.to_dict()
        g2 = UndirectedGraph.from_dict(data)
        self.assertEqual(set(g2.edges()), set(self.g.edges()))


if __name__ == "__main__":
    unittest.main()