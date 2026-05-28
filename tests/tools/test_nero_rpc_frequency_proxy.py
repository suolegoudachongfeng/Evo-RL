import unittest

from tools.nero_rpc_frequency_proxy import FrequencyStats, make_forwarder


class FakeTarget:
    def __init__(self):
        self.calls = []

    def servo_p_OL(self, robot_arm, pose, delta):
        self.calls.append(("servo_p_OL", robot_arm, pose, delta))
        return True


class NeroRpcFrequencyProxyTest(unittest.TestCase):
    def test_frequency_stats_groups_calls_by_method_and_robot_arm(self):
        stats = FrequencyStats(window_s=1.0)

        stats.record("servo_p_OL", 10.0, 10.002, ("right_robot", [0, 0, 0, 0, 0, 0], True))
        stats.record("servo_p_OL", 10.4, 10.405, ("right_robot", [0, 0, 0, 0, 0, 0], True))
        stats.record("right_robot_get_ee_pose", 10.5, 10.501, ())

        rows = stats.snapshot(now=10.9)

        self.assertEqual(rows[("servo_p_OL", "right_robot")]["count"], 2)
        self.assertEqual(rows[("servo_p_OL", "right_robot")]["hz"], 2.0)
        self.assertEqual(rows[("servo_p_OL", "right_robot")]["mean_latency_ms"], 3.5)
        self.assertEqual(rows[("right_robot_get_ee_pose", "-")]["count"], 1)

    def test_forwarder_records_and_forwards_exact_arguments(self):
        target = FakeTarget()
        stats = FrequencyStats(window_s=1.0)
        clock_values = iter([20.0, 20.003])

        forwarder = make_forwarder(
            target=target,
            method_name="servo_p_OL",
            stats=stats,
            clock=lambda: next(clock_values),
        )

        result = forwarder("right_robot", [1, 2, 3, 4, 5, 6], True)

        self.assertIs(result, True)
        self.assertEqual(target.calls, [("servo_p_OL", "right_robot", [1, 2, 3, 4, 5, 6], True)])
        rows = stats.snapshot(now=20.5)
        self.assertEqual(rows[("servo_p_OL", "right_robot")]["count"], 1)
        self.assertEqual(rows[("servo_p_OL", "right_robot")]["mean_latency_ms"], 3.0)


if __name__ == "__main__":
    unittest.main()
