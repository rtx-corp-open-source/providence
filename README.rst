Overview
--------

Purpose
^^^^^^^
This package is a library of functions and objects that can be used to work with populations of data.


Use Cases
^^^^^^^^^

- Defining independent and dependent variables
- Defining models that calculate dependent variables

Permitted Scope
^^^^^^^^^^^^^^^

- Parameter definition
- Parameter models
- General purpose validation routines (no tech data, may be removed in a future update)
- Plotting utilities (will be removed in a future update)

Unpermitted Scope
^^^^^^^^^^^^^^^^^

- Algorithms for predicting outputs
- Algorithms that work with models to determine information about the model
  - Ex 1. Sensitivity analysis
- Creation of statistical models
- Very common and more intricate capabilities that are used enough that they should be their own package.
  - Ex 1. Population-based optimization algorithms work with a population of design instances, but are far more complex than this package intends. Optimization belongs in a separate package that leverages the capabilities herein.
  - Ex 2. DOE generation is ubiquitous and, while it could fit into this package, may be better served elsewhere.

Location
^^^^^^^^

Code: https://github.devops.utc.com/Type-C-Lite/CARDS-parameters

Documentation, examples, contributing: https://cards.rtxdatascience.com/_static/cards_parameters/latest/index.html


History
^^^^^^^

In 2018, the CSE methods team had weekly 4-6 hour meetings where they argued over how to make this package as user friendly as possible. They succeeded.


Compliance
----------

License and Export Control
^^^^^^^^^^^^^^^^^^^^^^^^^^

The License and Required Marking (LICENSE_AND_REQUIRED_MARKING.rst) file provides the license and export control levels of the repository.

The Control Plan (CONTROL_PLAN.rst) contains requriements for participant compliance with Global Trade policy in this environment.

Unpublished data within this repository can be used by RTX users only. See the License and Required Marking (LICENSE_AND_REQUIRED_MARKING.rst) for more details and information on distribution.

`Alternate Means of Compliance (AMOC) Paper <https://devops.utc.com/-/media/Project/Corp/Corp-Intranet/Mattermost/DevOps-site2/Files/AMOC225_whitepaper_v1_2021-05-13.pdf?rev=cd7bf410327f42d1bff28fad89960b12&hash=0C423C181E00D0732AF0FA4F7061670C>`_

Restrictions
^^^^^^^^^^^^

**All users must understand and comply with the license and control plan files from the previous section.**

The following is a general list of **RESTRICTED** technology and software (via code, issues, PRs, etc). This is not a comprehensive list: the user is responsible for ensuring compliance with the control plan, and consulting their IP and GT points of contact when questions arise.

- Technical data related to:
  - A specific product with technology higher than 9E991
  - A military engine program
  - Material processing and/or parameters
  - Composites
  - Encryption
  - Real time signal processing
  - Electronic Engine Control Systems
- Any third party material, components, or code.
  - Any incorporation of third party code (e.g., open source, proprietary code) must **strictly** be by reference, and **must not be** included in the code base.
  - This restriction includes both in its complete and partial forms. **NO EXTERNAL CODE SHOULD EVER BE COPIED INTO A REPOSITORY.**
  - Consult IP legal for third party and/or open source license guidelines.
- Any personal data

The following guidelines **MUST** be adhered to:
- Do not contribute work performed under or relating to a Government contract to these packages.
- If an invention disclosure has been submitted, follow the instructions of the patent advisory board.
- If an invention disclosure is in progress or contemplated in the future, contact IP legal before contributing relevant information.

Escape Handling
^^^^^^^^^^^^^^^

If any of the above limitations are violated:

1. Follow your business unit's standard procedure for reporting incidents
2. Contact the repo maintainers and/or the RTX GitHub admins

Points of Contact
^^^^^^^^^^^^^^^^^

- RTX: Joe Calogero (joseph.calogero@rtx.com)
- PWA: Chris Ruoti (Christopher.ruoti@prattwhitney.com)
- PWC: Dan Fudge (Daniel.fudge@pwc.ca)
