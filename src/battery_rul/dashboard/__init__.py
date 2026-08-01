"""Streamlit dashboard for the battery digital twin.

The dashboard is a *view*. It holds no model logic, no feature engineering and
no thresholds of its own: everything comes from
:class:`~battery_rul.digital_twin.service.BatteryDigitalTwinService` or from the
HTTP client that wraps it. A dashboard that recomputes anything is a dashboard
that will eventually disagree with the API it is meant to display.
"""

from __future__ import annotations

__all__: list[str] = []
