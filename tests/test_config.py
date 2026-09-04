import pytest

from tinyserve.config import EngineConfig


def test_defaults_validate():
    assert EngineConfig().validate().kv_backend == "paged"


@pytest.mark.parametrize("bad", [dict(scheduling="magic"), dict(kv_backend="tape"),
                                 dict(max_batch_size=0), dict(block_size=0)])
def test_invalid_values_rejected(bad):
    with pytest.raises(ValueError):
        EngineConfig(**bad).validate()
