### Explanation of changes

Describe the contents of this merge request and the issues addressed.
Add screenshots if this helps your explanation.

### Motivation of changes

Motivate why the particular solution was chosen.

--------------------

## Merge checklist
See also [merge request guidelines](https://quantify-os.org/docs/quantify-core/dev/contributing.html#merge-request-guidelines)

- [ ] Merge request has been reviewed (in-depth by a knowledgeable contributor), and is approved by a project maintainer.
- [ ] New code is covered by unit tests (or N/A).
- [ ] New code is documented and docstrings use [numpydoc format](https://numpydoc.readthedocs.io/en/latest/format.html) (or N/A).
- [ ] New functionality: considered making private instead of extending public API (or N/A).
- [ ] Public API changed: added `@deprecated` and entry in [deprecated code suggestions](https://quantify-os.org/docs/quantify-scheduler/dev/examples/deprecated.html) (or N/A).
- [ ] Newly added/adjusted documentation and docstrings render properly (or N/A).
- [ ] Pipeline fix or dependency update: post in `#software-for-developers` channel to merge `main` back in or [update local packages](https://quantify-os.org/docs/quantify-scheduler/dev/user/installation.html#setting-up-for-local-development) (or N/A).
- [ ] Tested on hardware (or N/A).
- [ ] `CHANGELOG.md` and `AUTHORS.md` have been updated (or N/A).
- [ ] Performance tests: if changes can affect performance, trigger CI manually and evaluate results (or N/A).
- [ ] Windows tests in CI pipeline pass (manually triggered by maintainers before merging).
   - _Maintainers do not hit Auto-merge, we need to actively check as manual tests do not block pipeline_
---

For reference, the issues workflow is described in the [contribution guidelines](https://quantify-os.org/docs/quantify-core/dev/contributing.html#merge-requests-workflow).
