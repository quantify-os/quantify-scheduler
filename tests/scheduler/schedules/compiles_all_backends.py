# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


from quantify_scheduler.compilation import qcompile


class _CompilesAllBackends:
    """
    A mixin to be reused in the test classes of the same dir.

    Assumes a .sched attribute.
    """

    def test_compiles_qblox_backend(
        self, load_example_transmon_config, load_example_qblox_hardware_config
    ) -> None:
        # assert that files properly compile
        qcompile(
            self.sched,  # pylint: disable=no-member
            load_example_transmon_config(),
            load_example_qblox_hardware_config(),
        )

    def test_compiles_zi_backend(
        self, load_example_transmon_config, load_example_zhinst_hardware_config
    ) -> None:
        qcompile(
            self.sched,  # pylint: disable=no-member
            load_example_transmon_config(),
            load_example_zhinst_hardware_config(),
        )
