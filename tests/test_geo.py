import unittest

import numpy as np

from avito_improved.geo import distance_km, nearby_candidates


class GeoTests(unittest.TestCase):
    def test_distance_crosses_date_line(self):
        distance = distance_km(np.array([0.0]), np.array([-179.9]), (0.0, 179.9))
        self.assertAlmostEqual(distance[0], 22.24, places=1)

    def test_nearby_search_requires_coordinates_and_positive_match(self):
        latitude = np.array([0.0, 0.2, 1.0])
        longitude = np.zeros(3)
        scores = {'title': np.array([0.0, 2.0, 3.0])}
        np.testing.assert_array_equal(nearby_candidates(scores, latitude, longitude, (0.0, 0.0)), [1])
        self.assertEqual(len(nearby_candidates(scores, latitude, longitude, (None, None))), 0)
