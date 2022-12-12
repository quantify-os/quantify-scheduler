## Checklist for a new release

1. [ ] Review `CHANGELOG.md` and `AUTHORS.md` have been updated.  
1. [ ] Review `@deprecated` and `FutureWarning`s that can be cleaned up now.

1. CI pipeline:
    - [ ] Automated pipeline passes.
    - [ ] All `Test (py3.x, Windows, manual)` pass (trigger manually!).

1. [ ] Bump version and commit & push:
   ```bash
   VERSION_PART=patch # or minor, or major
   bump2version $VERSION_PART --config-file setup.cfg

   NEW_VERSION=$(python setup.py --version)
   echo $NEW_VERSION

   git add setup.py setup.cfg quantify_scheduler/__init__.py
   git commit -m "Bump to version $NEW_VERSION"
   git push
   ```
   
1. [ ] Commit pip frozen requirements for future reference:
    - Go to the `Test (py3.9, Linux)` pipeline job and download the `artifacts` (right side "Job artifacts" `-->` "Download").
    - Unzip, get the `frozen-requirements.txt`.
    - Paste it in `frozen_requirements` directory.
    - Rename it, commit & push:

      ```bash
      NEW_VERSION=$(python setup.py --version)
      echo $NEW_VERSION

      mv frozen_requirements/frozen-requirements.txt frozen_requirements/frozen-requirements-$NEW_VERSION.txt

      git add frozen_requirements/frozen-requirements-$NEW_VERSION.txt
      git commit -m "Add pip frozen requirements for $NEW_VERSION"
      git push
      ```

1. [ ] Create tag for bumped version:
    - Merge this MR into `main`.
    - Create tag via GitLab from `main` using the bumped version number (https://gitlab.com/quantify-os/quantify-scheduler/-/tags/new).
      - Do not add any description. 

    <!-- - Future TODO: finish automation of this step in `.gitlab-ci.yml`. -->
    <!-- 1. [ ] Run **one** of the major/minor/patch version bump (manual) jobs in the CI pipeline of the MR. -->
    <!--     - NB this can only be done after unix and windows test & docs jobs pass. -->

1. Read-the-Docs setup:
   - [ ] Enable docs build for the new tag [over here](https://readthedocs.com/projects/quantify-quantify-scheduler/versions/).
      - Configuration:
        - `Active`=True
        - `Hidden`=False
        - `Privacy Level`=Public
   - [ ] Change both the `Default version` and `Default branch` of the docs to the tag that was released [over here](https://readthedocs.com/dashboard/quantify-quantify-scheduler/advanced/). Hit Save!
   - [ ] Make sure the docs build and check on RTD.
      - Manually rebuild `latest` by hitting `Build version:` [over here](https://readthedocs.com/projects/quantify-quantify-scheduler/builds/).
      - Check both the `latest` and the new version links on RTD work by clicking through to Changelog (hit Ctrl+F5).

1. [ ] Create [new release on GitLab](https://gitlab.com/quantify-os/quantify-scheduler/-/releases).
    - Meaningful title
    - List of highlights followed by changelog.
    - Add a few images or animated GIFs showcasing the new exciting features.

1. Do TestPyPi release (also see https://adriaanrol.com/posts/pypi/):
    - [ ] Checkout the tag you just created:
       ```bash
       git fetch && git checkout $NEW_VERSION
       ```
    - [ ] Build package and upload to TestPyPi:
       ```bash
       # Always update first
       pip install --user --upgrade setuptools wheel

       # Clear dist/ directory
       rm dist/*

       # This creates several files in the dist/ directory
       python setup.py sdist bdist_wheel

       # If ^ runs without warnings you can upload to test.pypi.org
       python -m twine upload --repository testpypi dist/*
       ```
    - [ ] Install package in (test) env and validate (e.g., run a quick notebook). 
       ```bash
       pip install quantify-scheduler==x.x.x --extra-index-url=https://test.pypi.org/simple/
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

1. [ ] Release on PyPi and wait for it to become available (also see https://adriaanrol.com/posts/pypi/).
    ```bash
    twine upload dist/* 
    ```

1. [ ] Post the new release in Slack (`#software-for-users` and `#software-for-developers`).
    - PS Rockets are a must! 🚀🚀🚀
1. [ ] Inform the Quantify Marketing Team.
