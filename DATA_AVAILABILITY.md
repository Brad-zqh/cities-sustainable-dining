# Data availability

The author authorised public release of the processed inputs used to display
the V203 results on 5 October 2026. `processed_display_data/v203` contains
regional indicators, frozen statistics/intervals, display geometries and
anonymous public-housing plotting nodes. See [the exact scope](docs/PROCESSED_DISPLAY_DATA.md).
This is a processed display-data release, not an open deposit of every raw
dataset or a reproduction of the original data-acquisition and modelling chain.

Raw OpenRice records, reviews, menus and images; restaurant master/linkage
records; personal profiles; household microdata; routing OD pairs; credentials;
manuscripts; reviewer correspondence; and Word preparation/check scripts are
excluded. Individual bootstrap replicates and unrelated intermediate fields
are also excluded. Missing evidence and network-unavailable states are retained.

Code retains the repository MIT licence and fonts retain their own notices.
This release adds no CC BY or other blanket third-party data licence and no
invented DOI. Author release authorisation for these processed display files
does not establish new redistribution rights for the underlying platform,
census, FEHD or OSM source materials. Acquisition and third-party terms remain
separate; see [the existing notices](THIRD_PARTY_NOTICES.md).

The supplied V203 manuscript Figs. 12–19 and SI Fig. S9 are supported. Main
Figs. 1–11, SI Figs. S1–S8 and all other acquisition/model stages are outside
this display release. Result-table outputs are tagged as a V203 numerical
snapshot, while SI wording, numbering and layout may be revised separately.

`manifests/source_inventory.json` describes a larger historical private bundle.
Its metadata is not the public data contract. The release-specific contract is
`processed_display_data/v203/manifest.json`, accompanied by SHA256SUMS and a
field dictionary. Hashes establish file identity, not source ownership or
independent scientific validation.
