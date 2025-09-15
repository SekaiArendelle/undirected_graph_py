"""
Unit-tests for the `UndirectedGraph` container.
"""

import unittest
from undirected_graph import (
    UndirectedGraph,
    NodeExistsError,
    NodeNotExistsError,
    EdgeNotExistsError,
)


class TestUndirectedGraph(unittest.TestCase):
    """Test cases for every public method of UndirectedGraph."""

    def _populate_triangle(self) -> UndirectedGraph:
        """Add the edges of a 3-cycle (triangle).

        Returns:
            UndirectedGraph: A graph with three nodes connected in a triangle.
        """
        g = UndirectedGraph()
        g.add_node(1)
        g.add_node(2)
        g.add_node(3)

        g.construct_edge(1, 2, 1)
        g.construct_edge(2, 3, 1)
        g.construct_edge(3, 1, 1)

        return g


    def test_empty_graph(self):
        """Test that a newly created graph is empty."""
        g = UndirectedGraph()
        self.assertEqual(len(g), 0)
        self.assertEqual(list(g.nodes()), [])
        self.assertEqual(list(g.edges()), [])

    def test_contains(self):
        """Test the __contains__ method of UndirectedGraph."""
        g = UndirectedGraph()

        g.add_node("A")
        self.assertIn("A", g)
        self.assertNotIn("B", g)

    def test_len_len(self):
        """Test that the length of the graph corresponds to the number of nodes."""
        g = UndirectedGraph()
        g.add_node("X")
        g.add_node("Y")
        g.construct_edge("X", "Y", 1)
        self.assertEqual(len(g), 2)


    def test_add_node_idempotent(self):
        """Test that adding the same node twice raises NodeExistsError."""
        g = UndirectedGraph()

        g.add_node("A")
        try:
            g.add_node("A")
        except NodeExistsError:
            pass
        else:
            assert False

    def test_add_edge_creates_nodes(self):
        """Test that adding an edge also adds the nodes if they don't exist."""
        g = UndirectedGraph()

        g.add_node(1)
        g.add_node(2)
        g.construct_edge(1, 2, 1)

        self.assertEqual(len(g), 2)
        self.assertEqual(g.degree(1), 1)
        self.assertEqual(g.degree(2), 1)

    def test_add_edge_idempotent(self):
        """Test that adding the same edge twice updates the weight but doesn't duplicate."""
        g = UndirectedGraph()

        g.add_node(1)
        g.add_node(2)
        g.construct_edge(1, 2, 1)
        g.assign_edge(1, 2, 2)

        self.assertEqual(g.degree(1), 1)
        self.assertEqual(len(g), 2)

    def test_remove_node(self):
        """Test removing a node and its associated edges from the graph."""
        g = self._populate_triangle()

        g.remove_node(2)
        self.assertNotIn(2, g)
        self.assertEqual(g.degree(1), 1)  # only 3 remains
        self.assertEqual(g.degree(3), 1)

    def test_remove_node_missing(self):
        """Test that removing a non-existent node raises NodeNotExistsError."""
        g = UndirectedGraph()
        try:
            g.remove_node("ghost")
        except NodeNotExistsError:
            pass
        else:
            assert False

    def test_remove_edge(self):
        """Test removing an edge between two nodes."""
        g = UndirectedGraph()
        g.add_node(1)
        g.add_node(2)
        g.construct_edge(1, 2, 1)
        g.remove_edge(1, 2)
        self.assertEqual(g.degree(1), 0)
        self.assertEqual(g.degree(2), 0)

    def test_remove_edge_missing(self):
        """Test that removing a non-existent edge raises EdgeNotExistsError."""
        g = UndirectedGraph()

        g.add_node(1)
        g.add_node(2)
        try:
            g.remove_edge(1, 2)
        except EdgeNotExistsError:
            pass
        else:
            assert False

    def test_neighbors(self):
        """Test retrieving the neighbors of a node."""
        g = self._populate_triangle()
        self.assertEqual(set(g.neighbors(1)), {2, 3})

    def test_neighbors_missing(self):
        """Test that accessing neighbors of a non-existent node raises KeyError."""
        g = UndirectedGraph()

        with self.assertRaises(KeyError):
            list(g.neighbors("no-such-node"))

    def test_degree(self):
        """Test retrieving the degree of a node."""
        g = self._populate_triangle()
        self.assertEqual(g.degree(1), 2)

    def test_edges_unique(self):
        """Test that edges are unique and not duplicated."""
        g = self._populate_triangle()
        edges = list(g.edges())
        self.assertEqual(len(edges), 3)
        # Ensure no duplicates such as (1,2) and (2,1)
        self.assertEqual(len(set(edges)), 3)

    def test_copy(self):
        """Test creating a copy of the graph."""
        g = UndirectedGraph()
        g.add_node("A")
        g.add_node("B")
        g.construct_edge("A", "B", 1)

        g2 = g.copy()
        g2.remove_node("A")
        self.assertIn("A", g)
        self.assertNotIn("A", g2)

    def test_clear(self):
        """Test clearing all nodes and edges from the graph."""
        g = self._populate_triangle()
        g.clear()
        self.assertEqual(len(g), 0)

    def test_swap(self):
        """Test swapping the contents of two graphs."""
        g = UndirectedGraph()
        g2 = self._populate_triangle()
        g.swap(g2)
        self.assertTrue(g2.empty())
        self.assertFalse(g.empty())


if __name__ == "__main__":
    unittest.main()
