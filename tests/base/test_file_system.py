import pytest

from not_again_ai.base.file_system import readable_size


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (0, "0.00 B"),
        (523, "523.00 B"),
        (2048, "2.00 KB"),
        (5242880, "5.00 MB"),
        (10737418240, "10.00 GB"),
        (1099511627776, "1.00 TB"),
    ],
)
def test_human_readable_size(size: float, expected: str) -> None:
    assert readable_size(size) == expected, f"Failed for {size} bytes"
