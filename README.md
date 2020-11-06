<div align="center"><img src="https://raw.githubusercontent.com/PlasmaPy/PlasmaPy-logo/master/exports/with-text-dark.png" width="600"/></div>

# PlasmaPy

[![PyPI version](https://badge.fury.io/py/plasmapy.svg)](https://badge.fury.io/py/plasmapy)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](./LICENSE.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1436011.svg)](https://doi.org/10.5281/zenodo.1436011)

[![Build Status](https://travis-ci.org/PlasmaPy/PlasmaPy.svg?branch=master)](https://travis-ci.org/PlasmaPy/PlasmaPy)
[![Build Status](https://dev.azure.com/plasmapy/PlasmaPy/_apis/build/status/PlasmaPy.PlasmaPy?branchName=master)](https://dev.azure.com/plasmapy/PlasmaPy/_build/latest?definitionId=2&branchName=master)
[![Build status](https://ci.appveyor.com/api/projects/status/hbduy62sqrvy8rn7?svg=true)](https://ci.appveyor.com/project/namurphy/plasmapy)
[![codecov](https://codecov.io/gh/PlasmaPy/PlasmaPy/branch/master/graph/badge.svg)](https://codecov.io/gh/PlasmaPy/PlasmaPy)
[![Documentation Status](https://readthedocs.org/projects/plasmapy/badge/?version=latest)](http://plasmapy.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/PlasmaPy/PlasmaPy/master?filepath=plasmapy%2Fexamples)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

[![Matrix](https://matrix.to/img/matrix-badge.svg)](https://app.element.io/#/room/#plasmapy:openastronomy.org)
[![Gitter (bridged to Matrix)](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/PlasmaPy/Lobby)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)
[![Open Source Helpers](https://www.codetriage.com/plasmapy/plasmapy/badges/users.svg)](https://www.codetriage.com/plasmapy/plasmapy)
[![All Contributors](https://img.shields.io/badge/all_contributors-0-orange.svg?style=flat-square)](#contributors-)

PlasmaPy is an open source community developed Python 3.6+ package for
plasma physics in the early stages of development.  PlasmaPy intends to
be for plasmas what [Astropy](https://github.com/astropy/astropy) is for
astronomy — a collection of functionality commonly used and shared
between plasma physicists and researchers globally, running within and
leveraging the open source scientific Python ecosystem.  The goals of
this project are more thoroughly described in our [vision
statement](http://docs.plasmapy.org/en/stable/about/vision_statement.html)
and [this recent reference](https://doi.org/10.5281/zenodo.1238132).
We are in the process of writing [online
documentation](http://docs.plasmapy.org/en/latest/).

We created a guide on [contributing to PlasmaPy](http://docs.plasmapy.org/en/stable/CONTRIBUTING.html)
and have a [Code of Conduct](http://docs.plasmapy.org/en/stable/CODE_OF_CONDUCT.html).
New contributors are very welcome!

# Feedback and communication

## [Matrix chat](https://app.element.io/#/room/#plasmapy:openastronomy.org)

If you have any questions, the quickest way to get a response is to ask
on our
[Matrix](https://app.element.io/#/room/#plasmapy:openastronomy.org)/[Gitter](https://gitter.im/PlasmaPy/Lobby)
channel. Both of these are the same chat channel; Gitter uses a bridge to link the two.

## [Discourse room](https://plasmapy.discourse.group/)

We have recently created a [PlasmaPy Discourse
group](https://plasmapy.discourse.group/) in order to allow threaded
public discussions on a variety of topics.  This group is a great
place to suggest ideas, bring up discussion topics, and ask questions.

## [Mailing list](https://groups.google.com/forum/#!forum/plasmapy)

We also have a [mailing list](https://groups.google.com/forum/#!forum/plasmapy)
that serves as a less volatile discussion forum.

## [Suggestion box](https://docs.google.com/forms/d/e/1FAIpQLSdT3O5iHZrLJRuavFyzoR23PGy0Prfzx2SQOcwJGWtvHyT2lw/viewform?usp=sf_link)

We have
[a suggestion box](https://docs.google.com/forms/d/e/1FAIpQLSdT3O5iHZrLJRuavFyzoR23PGy0Prfzx2SQOcwJGWtvHyT2lw/viewform?usp=sf_link)
if you would like to (optionally anonymously) suggest
a feature/topic for consideration. These will be reposted on the mailing list
or directly in GitHub issues, as appropriate, for further discussion.

## [Weekly](https://calendar.google.com/calendar?cid=bzVsb3ZkcW0zaWxsam00ZTlrMDd2cmw5bWdAZ3JvdXAuY2FsZW5kYXIuZ29vZ2xlLmNvbQ) [community meetings](https://meet.jit.si/plasmapy)

We have weekly community meetings in the
[PlasmaPy room on Jitsi](https://meet.jit.si/plasmapy).
The schedule of our community meetings is on our [calendar](https://calendar.google.com/calendar?cid=bzVsb3ZkcW0zaWxsam00ZTlrMDd2cmw5bWdAZ3JvdXAuY2FsZW5kYXIuZ29vZ2xlLmNvbQ), and you may access the [minutes and
agendas](https://drive.google.com/drive/folders/0ByPG8nie6fTPV1FQUEkzMTgtRTg?usp=sharing).
Any last minute changes will be discussed on
[Matrix](https://app.element.io/app/#/room/#plasmapy:openastronomy.org).
As of April 2020, our meetings are on Tuesdays at
[18:00 UTC](http://time.unitarium.com/utc/6pm).
Come discuss plasma software with us!

# Installation

You can get PlasmaPy from pip via `pip install plasmapy`. To contribute
to the package, check out [our instructions on installing PlasmaPy from
source](http://docs.plasmapy.org/en/stable/install.html#building-and-installing-from-source-code).

You can also get PlasmaPy from `conda` via `conda install -c conda-forge plasmapy`.

Like most scientific Python packages, PlasmaPy probably runs better on the
[Anaconda distribution](https://www.anaconda.com/downloads).

PlasmaPy requires Python 3.6+ and is [not compatible with
Python 2](https://pythonclock.org/).

# License

PlasmaPy is licensed under a 3-clause BSD license with added protections
against software patents — see the [``LICENSE.md``](LICENSE.md) file in
the top-level directory.

# Citing PlasmaPy

If you use PlasmaPy in a research publication, we ask that you cite the
Zenodo record for the specific version of PlasmaPy that you used.
Please see the [``docs/about/citation.rst``](./docs/about/citation.rst)
file for more detailed instructions on how to cite PlasmaPy.

# Acknowledgements

Early development on PlasmaPy was supported in part by the U.S.
Department of Energy, the Smithsonian Institution, and Google Summer of
Code.  Ongoing PlasmaPy development is being supported through a
collaborative award from the U.S. National Science Foundation's
Cyberinfrastructure for Sustained Scientific Innovation program and a
NASA HDEE grant.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/colbych"><img src="https://avatars3.githubusercontent.com/u/3066366?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Colby Haggerty</b></sub></a><br /><a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=colbych" title="Code">💻</a> <a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=colbych" title="Documentation">📖</a></td>
    <td align="center"><a href="https://www.davidstansby.com/"><img src="https://avatars0.githubusercontent.com/u/6197628?v=4?s=100" width="100px;" alt=""/><br /><sub><b>David Stansby</b></sub></a><br /><a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=dstansby" title="Code">💻</a> <a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=dstansby" title="Documentation">📖</a> <a href="#infra-dstansby" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=dstansby" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://stanczakdominik.github.io/"><img src="https://avatars0.githubusercontent.com/u/11289391?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Dominik Stańczak</b></sub></a><br /><a href="#question-StanczakDominik" title="Answering Questions">💬</a> <a href="#blog-StanczakDominik" title="Blogposts">📝</a> <a href="https://github.com/PlasmaPy/PlasmaPy/issues?q=author%3AStanczakDominik" title="Bug reports">🐛</a> <a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=StanczakDominik" title="Code">💻</a> <a href="#design-StanczakDominik" title="Design">🎨</a> <a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=StanczakDominik" title="Documentation">📖</a> <a href="#example-StanczakDominik" title="Examples">💡</a> <a href="#ideas-StanczakDominik" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-StanczakDominik" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-StanczakDominik" title="Maintenance">🚧</a> <a href="#platform-StanczakDominik" title="Packaging/porting to new platform">📦</a> <a href="#research-StanczakDominik" title="Research">🔬</a> <a href="https://github.com/PlasmaPy/PlasmaPy/pulls?q=is%3Apr+reviewed-by%3AStanczakDominik" title="Reviewed Pull Requests">👀</a> <a href="#talk-StanczakDominik" title="Talks">📢</a> <a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=StanczakDominik" title="Tests">⚠️</a> <a href="#tool-StanczakDominik" title="Tools">🔧</a> <a href="#tutorial-StanczakDominik" title="Tutorials">✅</a></td>
    <td align="center"><a href="https://github.com/SolarDrew"><img src="https://avatars2.githubusercontent.com/u/1914702?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Drew Leonard</b></sub></a><br /><a href="#question-SolarDrew" title="Answering Questions">💬</a> <a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=SolarDrew" title="Code">💻</a> <a href="#design-SolarDrew" title="Design">🎨</a> <a href="#ideas-SolarDrew" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-SolarDrew" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-SolarDrew" title="Maintenance">🚧</a> <a href="#mentoring-SolarDrew" title="Mentoring">🧑‍🏫</a> <a href="https://github.com/PlasmaPy/PlasmaPy/pulls?q=is%3Apr+reviewed-by%3ASolarDrew" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=SolarDrew" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/rocco8773"><img src="https://avatars1.githubusercontent.com/u/29869348?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Erik Everson</b></sub></a><br /><a href="#question-rocco8773" title="Answering Questions">💬</a> <a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=rocco8773" title="Code">💻</a> <a href="#design-rocco8773" title="Design">🎨</a> <a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=rocco8773" title="Documentation">📖</a> <a href="#example-rocco8773" title="Examples">💡</a> <a href="#ideas-rocco8773" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-rocco8773" title="Maintenance">🚧</a> <a href="#mentoring-rocco8773" title="Mentoring">🧑‍🏫</a> <a href="#projectManagement-rocco8773" title="Project Management">📆</a> <a href="#research-rocco8773" title="Research">🔬</a> <a href="https://github.com/PlasmaPy/PlasmaPy/pulls?q=is%3Apr+reviewed-by%3Arocco8773" title="Reviewed Pull Requests">👀</a> <a href="#tutorial-rocco8773" title="Tutorials">✅</a> <a href="#video-rocco8773" title="Videos">📹</a></td>
    <td align="center"><a href="https://github.com/KhalilBryant"><img src="https://avatars3.githubusercontent.com/u/35078079?v=4?s=100" width="100px;" alt=""/><br /><sub><b>KhalilBryant</b></sub></a><br /><a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=KhalilBryant" title="Code">💻</a></td>
    <td align="center"><a href="https://www.plasmapy.org/"><img src="https://avatars0.githubusercontent.com/u/8931994?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Nick Murphy</b></sub></a><br /><a href="#a11y-namurphy" title="Accessibility">️️️️♿️</a> <a href="#question-namurphy" title="Answering Questions">💬</a> <a href="https://github.com/PlasmaPy/PlasmaPy/issues?q=author%3Anamurphy" title="Bug reports">🐛</a> <a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=namurphy" title="Code">💻</a> <a href="#content-namurphy" title="Content">🖋</a> <a href="#data-namurphy" title="Data">🔣</a> <a href="#design-namurphy" title="Design">🎨</a> <a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=namurphy" title="Documentation">📖</a> <a href="#eventOrganizing-namurphy" title="Event Organizing">📋</a> <a href="#example-namurphy" title="Examples">💡</a> <a href="#fundingFinding-namurphy" title="Funding Finding">🔍</a> <a href="#ideas-namurphy" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-namurphy" title="Maintenance">🚧</a> <a href="#mentoring-namurphy" title="Mentoring">🧑‍🏫</a> <a href="#projectManagement-namurphy" title="Project Management">📆</a> <a href="#research-namurphy" title="Research">🔬</a> <a href="https://github.com/PlasmaPy/PlasmaPy/pulls?q=is%3Apr+reviewed-by%3Anamurphy" title="Reviewed Pull Requests">👀</a> <a href="#talk-namurphy" title="Talks">📢</a> <a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=namurphy" title="Tests">⚠️</a> <a href="#tutorial-namurphy" title="Tutorials">✅</a> <a href="#video-namurphy" title="Videos">📹</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/lemmatum"><img src="https://avatars0.githubusercontent.com/u/28945309?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Pawel Kozlowski</b></sub></a><br /><a href="#question-lemmatum" title="Answering Questions">💬</a> <a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=lemmatum" title="Code">💻</a> <a href="#design-lemmatum" title="Design">🎨</a></td>
    <td align="center"><a href="http://www.physics.ucla.edu/~pheuer/"><img src="https://avatars0.githubusercontent.com/u/32618747?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Peter Heuer</b></sub></a><br /><a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=pheuer" title="Code">💻</a> <a href="#design-pheuer" title="Design">🎨</a> <a href="#example-pheuer" title="Examples">💡</a> <a href="#ideas-pheuer" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://ritiek.github.io/"><img src="https://avatars1.githubusercontent.com/u/20314742?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ritiek Malhotra</b></sub></a><br /><a href="#question-ritiek" title="Answering Questions">💬</a> <a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=ritiek" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/RoberTnf"><img src="https://avatars1.githubusercontent.com/u/6000936?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Roberto Díaz Pérez</b></sub></a><br /><a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=RoberTnf" title="Code">💻</a></td>
    <td align="center"><a href="http://lostechies.com/seanchambers/"><img src="https://avatars3.githubusercontent.com/u/31563?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Sean Chambers</b></sub></a><br /><a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=schambers" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/thomasjpfan"><img src="https://avatars2.githubusercontent.com/u/5402633?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Thomas J. Fan</b></sub></a><br /><a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=thomasjpfan" title="Code">💻</a> <a href="#platform-thomasjpfan" title="Packaging/porting to new platform">📦</a></td>
    <td align="center"><a href="https://github.com/jasperbeckers"><img src="https://avatars0.githubusercontent.com/u/28015654?v=4?s=100" width="100px;" alt=""/><br /><sub><b>jasperbeckers</b></sub></a><br /><a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=jasperbeckers" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/samurai688"><img src="https://avatars1.githubusercontent.com/u/12175315?v=4?s=100" width="100px;" alt=""/><br /><sub><b>samurai688</b></sub></a><br /><a href="https://github.com/PlasmaPy/PlasmaPy/commits?author=samurai688" title="Code">💻</a> <a href="#design-samurai688" title="Design">🎨</a> <a href="#research-samurai688" title="Research">🔬</a> <a href="https://github.com/PlasmaPy/PlasmaPy/pulls?q=is%3Apr+reviewed-by%3Asamurai688" title="Reviewed Pull Requests">👀</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
