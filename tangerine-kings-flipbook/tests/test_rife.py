from lib.rife import expected_output_count


def test_expected_output_count_3x_interpolation():
    assert expected_output_count(0, 3) == 0
    assert expected_output_count(1, 3) == 1
    assert expected_output_count(2, 3) == 4
    assert expected_output_count(1582, 3) == 4744


def test_expected_output_count_2x():
    assert expected_output_count(10, 2) == 19
