# Data availability and release gates

This release contains code and source-file metadata, **not empirical data**.
The local figure bundle contains 174 source files. Its file hashes and tabular
schemas are documented in `manifests/source_inventory.json`; these metadata
allow subsequent verification but cannot reproduce the numerical figures alone.

| Material | Current access route | What is still required |
| --- | --- | --- |
| Author-written plotting/estimator code | Public repository | Ongoing code review and versioned fixes |
| Platform-derived LSBG/DCCA counts, scores and bootstrap tables | Withheld pending redistribution clearance | Record applicable authorization or licence, including permitted aggregation/resolution |
| Census, FEHD and public-housing spatial derivatives | Withheld pending source-specific review | Original dataset identifiers, versions, licence terms and attribution |
| Basemap tile cache and old preview imagery | Not included; not needed by current figures | No basemap is used in current figures |
| Raw reviews, menus, photographs, restaurant identifiers and OD pairs | Not included | Separate rights/privacy basis; no blanket redistribution claim |
| Conceptual/LLM workflow illustrations | Not part of numerical reproduction | Authorship/provenance and approved manuscript exports |

The current [OpenRice service terms](https://www.openrice.com/info/tnc/OR-terms-en.html)
contain restrictions on copying/extraction and third-party rights. They do not
establish the historical terms or any separate authorization applicable to the
study. Existing lawful access and permission to redistribute must be documented
separately; neither is inferred here. This checklist is not a legal opinion or
an allegation of misconduct.

The rights review is unresolved, not a statement that clearance has been denied.
The repository will not declare CC BY for third-party material without a basis.
The current code release is therefore an **incomplete data-release milestone**.

## Suggested manuscript wording while clearance is pending

The visualization code and shared analytical functions are available in the
accompanying code repository. Redistribution of the empirical source-data bundle
is pending source-specific permission and licence checks. Accordingly, the
current public release does not yet support independent numerical reproduction
of all figures. Restricted raw platform records and origin–destination pairs
are not included.

Do not replace this with “all code and data are publicly available” until the
cleared empirical bundle is actually deposited and independently downloaded.
