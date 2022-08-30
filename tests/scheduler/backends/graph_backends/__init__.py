"""
Module containing tests for the different (graph-based) backends for quantify scheduler.
The idea is that we follow a standard template on how we test the compilation for the
different backends so that we ensure they all support (to the extent possible) the same
functionality.

To do this, based on a set of standard schedules (fixtures), we verify if the different
backends compile all of these programs with a suitable standard configuration file.
After that, we make specific tests to verify the output at the end of the compilation
graph.

Separate to these, there are also tests that test the functionality of specific nodes.
These could/should be separated from the integration-like tests that are done here.

Note that all I write here could change in the future, to facilitate refactor we should
attempt to separate the testing of requirements/functionality from that of the
implementation.
"""
