# ppBang

Statistical performance calculation for *osu!*

More details on how this actually works are contained in `pp!.pdf`.

## Getting started

Requires a relevant [data dump](https://data.ppy.sh/) folder to be present in the working directory, with files in the
SQLite database format. The date, type, and game mode of this dump are set in `comparison_graph.py` by `data_dump_date`,
`data_dump_type`, and `game_mode` respectively.

### Comparison graph

A *comparison graph* can be generated by running `comparison_graph.py`, and is written to the file
`comparison_graph.gt`.
This is a weighted oriented graph with edges representing pairwise comparisons between `(beatmap_id, enabled_mods)`
tuples, and their weights representing a relative imbalance of difficulty.

### Map difficulty

The `difficulty.py` module contains functions for calculating a *difficulty* value for each
`(beatmap_id, enabled_mods)` tuple, using the graph stored in `comparison_graph.gt`.
