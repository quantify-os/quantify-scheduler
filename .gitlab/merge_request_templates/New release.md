## Checklist for a new release

1. [ ] Review that `AUTHORS.md` has been updated.
1. [ ] Review `@deprecated` and `FutureWarnings` that can be cleaned up now.

1. [ ] Add compatibility info to `README.md`, extract the versions from `pyproject.toml` and https://pypi.org/project/qblox-instruments/ (Qblox Cluster firmware). e.g.:
      ```
      ### Compatibility info

      - Qblox: `qblox-instruments==x.x.x` (Cluster firmware vx.x.x)
      - ZI:    `zhinst==x.x.x` `zhinst-qcodes==x.x.x` `zhinst-toolkit==x.x.x`
      ```

1. CI pipeline:
    - [ ] Add "Release" label to trigger the changelog update pipeline job. (make sure that "vX.Y.Z" is somewhere in the merge request title)
    - [ ] Automated pipeline passes.
    - [ ] All `Test (py3.x, Windows, manual)` pass (trigger manually!).


1. [ ] Create tag for bumped version:
    - Merge this MR into `main`.
    - Switch to `main` branch.
    - Create and push an **annotated** tag `vX.Y.Z` pointing to the merge commit:

      ```bash
      echo $NEW_VERSION

      git tag -a "v${NEW_VERSION}"  # Note: should be vX.Y.Z, not X.Y.Z
      # You will be prompted for a tag description: `Release vX.Y.Z`
      git push origin "v${NEW_VERSION}"
      ```

1. [ ] Push an empty commit to `main` with message `Start development of vX.Y.Z+1.dev`.
1. [ ] **Create** and **push** (see steps above) an **annotated** tag `vX.Y.Z+1.dev` pointing to the commit above.  Commit annotation: `Start development of vX.Y.Z+1`.

1. When `Release to test.pypi.org` job of the tag pipeline succeeds:
    - [ ] Install package in (test) env and validate (e.g., run a quick notebook).
       ```bash
       pip install quantify-core==x.x.x --extra-index-url=https://test.pypi.org/simple/
       ```
       - _(For creating test env)_
         ```bash
         ENV_NAME=qtest # Adjust
         PY_VER=3.8
         DISPLAY_NAME="Python $PY_VER Quantify Test Env" && echo $DISPLAY_NAME # Adjust

         conda create --name $ENV_NAME python=$PY_VER
         conda activate $ENV_NAME
         conda install -c conda-forge jupyterlab
         python -m ipykernel install --user --name=$ENV_NAME --display-name="$DISPLAY_NAME"
         ```

1. [ ] Release on PyPi by triggering manual `Release to pypi.org` job and wait till it succeeds.
1. [ ] Post the new release in Slack (`#software-for-users` and `#software-for-developers`).
    - PS Rockets are a must! 🚀🚀🚀
1. [ ] Inform the Quantify Marketing Team.



