import pytest
from quantify_scheduler.backends import zhinst_backend

zhinst_backend.ZhinstBackend()


@pytest.mark.xfail(reason="NotImplemented")
def test_graph_compilation_idential_to_old_compile():
    raise NotImplementedError
