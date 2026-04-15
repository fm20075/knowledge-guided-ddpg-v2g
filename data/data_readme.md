# Dataset Description

The simulation data was generated based on the Hangzhou Xihu District parking lot scenario.

- `time_slots`: 0–23 hours.
- `grid_base_load`: Daily base load curve (kW).
- `ev_data`: List of EV records, each containing:
  - `ev_id`
  - `access_time` (hour)
  - `leave_time` (hour)
  - `init_soc` (0–1)
  - `target_soc` (0–1)
  - `current_soc` (initialized as init_soc)

To replace with real data, ensure the same structure and run the pipeline without modification.