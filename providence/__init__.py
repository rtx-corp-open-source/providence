"""
This is the core module of Providence.
We do no auxilliary adjustments to __all__ as we do in some submodules.

The codebase is function-first, rather than class/object first.
It is our experience that use of classes tends to encourage overuse of classes.
Much data-oriented work (e.g. deep learning) is more intuitively 'functional', which makes an object-orientation seem forced
or awkward for the library user. Yet, certain systems leverage objects internally for user benefit.
This is the approach of providence.

Please see the READMEs for more.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""

from providence._version import __version__
