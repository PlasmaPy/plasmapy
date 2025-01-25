[changelog guide]: https://docs.plasmapy.org/en/latest/contributing/changelog_guide.html
[coding guide]: https://docs.plasmapy.org/en/latest/contributing/coding_guide.html
[doc guide]: https://docs.plasmapy.org/en/latest/contributing/doc_guide.html
[testing guide]: https://docs.plasmapy.org/en/latest/contributing/testing_guide.html

Thank you for contributing to PlasmaPy! The project's future depends deeply on contributors like you, so we deeply appreciate it! 🌱 The following checklist will be used by the code reviewer to help guide the code review process.

- Overall
  - [ ] Does the PR do what it intends to do?
  - [ ] Except for very minor changes, is a changelog entry included and consistent with the [changelog guide]?
  - [ ] Are the continuous integration checks passing? (Most linter problems can be automagically fixed by commenting on this PR with `pre-commit.ci autofix`.)
- Code
  - [ ] Is new/updated code understandable and consistent with the [coding guide]?
  - [ ] Are there ways to greatly simplify the implementation?
  - [ ] Are there any large functions that should be split up into shorter functions?
- Tests
  - [ ] Are tests added/updated as required, and consistent with the [testing guide]?
  - [ ] Are the tests understandable?
  - [ ] Do the tests cover all important cases?
- Docs
  - [ ] Are docs added/updated as required, and consistent with the [doc guide]?
  - [ ] Are the docs understandable?
  - [ ] Do the docs show up correctly in the preview, including Jupyter notebooks?
