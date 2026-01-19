import unittest

from crunch_synth.quarantine import Quarantine



class TestQuarantine(unittest.TestCase):

    def test_add_to_quarantine_orders_correctly(self):
        q = Quarantine()

        q.add(time=10, value="A", horizon=5, steps=[1])   # release = 15
        q.add(time=5, value="B", horizon=20, steps=[1])  # release = 25
        q.add(time=12, value="C", horizon=3, steps=[1])   # release = 15

        assert q.quarantine == [
            (15, "A", [1]),
            (15, "C", [1]),
            (25, "B", [1]),
        ]

    def test_pop_returns_empty_when_no_item_ready(self):
        q = Quarantine()

        q.add(time=10, value="A", horizon=5, steps=[1])    # release = 15
        q.add(time=20, value="B", horizon=10, steps=[1])   # release = 30

        result = q.pop(current_time=14)

        assert result == []
        assert q.quarantine == [(15, "A", [1]), (30, "B", [1])]

    def test_pop_returns_one_item(self):
        q = Quarantine()

        q.add(time=10, value="A", horizon=5, steps=[1])   # release = 15
        q.add(time=20, value="B", horizon=10, steps=[1])  # release = 30

        result = q.pop(current_time=15)

        assert result == [(15, "A", [1])]
        assert q.quarantine == [(30, "B", [1])]

    def test_pop_returns_multiple_items(self):
        q = Quarantine()

        q.add(time=10, value="A", horizon=5, steps=[1])   # 15
        q.add(time=11, value="B", horizon=4, steps=[1])   # 15
        q.add(time=20, value="C", horizon=10, steps=[1])  # 30

        result = q.pop(current_time=20)

        assert result == [(15, "A", [1]), (15, "B", [1])]
        assert q.quarantine == [(30, "C", [1])]

    def test_pop_clears_all_items_if_all_ready(self):
        q = Quarantine()

        q.add(time=1, value="A", horizon=1, steps=[1])  # 2
        q.add(time=2, value="B", horizon=1, steps=[1])  # 3

        result = q.pop(current_time=10)

        assert result == [(2, "A", [1]), (3, "B", [1])]
        assert q.quarantine == []

    def test_pop_multiple_calls_progressively(self):
        q = Quarantine()

        q.add(time=10, value="A", horizon=5, steps=[1])   # 15
        q.add(time=12, value="B", horizon=3, steps=[1])   # 15
        q.add(time=20, value="C", horizon=10, steps=[1])  # 30

        # First pop: up to t=15
        first = q.pop(current_time=15)
        assert first == [(15, "A", [1]), (15, "B", [1])]
        assert q.quarantine == [(30, "C", [1])]

        # Second pop: nothing new
        second = q.pop(current_time=29)
        assert second == []
        assert q.quarantine == [(30, "C", [1])]

        # Third pop: C is now ready
        third = q.pop(current_time=30)
        assert third == [(30, "C", [1])]
        assert q.quarantine == []

if __name__ == "__main__":
    unittest.main()