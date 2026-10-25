# Export verification

All31original regression tests passed after relocating the planner into the packaged workspace layout. Includes real direct/staged trace parsing, slot and temporary-buffer lifetime, source eviction scheduling, missing calibration rejection, same-budget payloads and tree-initial profile export.

Portable entry point reproduced direct8:573Q3/195Q2,ratio1.0199878637677475,within_tolerance. Staged8:768Q2,ratio1.3840713399897924,not_reached_by_coordinate_search. No phone inference, new calibration or quality evaluation was run as part of this export.

The extra wrapper and test launcher are packaging entry points; original algorithm modules and cost tables are copied unchanged. Current CPU8/no-mid-logits performance is not retroactively fitted into them.
