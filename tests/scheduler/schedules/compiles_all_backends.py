# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import pprint

from quantify_scheduler.backends import SerialCompiler


class _CompilesAllBackends:
    """
    A mixin to be reused in the test classes of the same dir.

    Assumes a .sched attribute.
    """

    def test_compiles_qblox_backend(
        self, compile_config_basic_transmon_qblox_hardware
    ) -> None:
        # assert that files properly compile
        compilation_config = compile_config_basic_transmon_qblox_hardware
        compiler = SerialCompiler(name="compiler")
        try:
            compiler.compile(schedule=self.uncomp_sched, config=compilation_config)
        except ValueError:
            pprint.pprint(compilation_config.dict())
            raise

    def test_compiles_zi_backend(
        self, compile_config_basic_transmon_zhinst_hardware
    ) -> None:
        compilation_config = compile_config_basic_transmon_zhinst_hardware
        compiler = SerialCompiler(name="compiler")
        compiler.compile(schedule=self.uncomp_sched, config=compilation_config)
