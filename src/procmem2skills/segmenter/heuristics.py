from __future__ import annotations

from procmem2skills.models import BoundaryReason, Event, Segment, Trajectory


def segment_trajectory(trajectory: Trajectory, max_events_per_segment: int = 8) -> list[Segment]:
    if not trajectory.events:
        return []

    segments: list[Segment] = []
    current_events: list[Event] = []
    current_reasons: list[BoundaryReason] = []
    start_step = trajectory.events[0].step_id

    def flush(end_step: int) -> None:
        if not current_events:
            return
        segment_id = f"{trajectory.episode_id}-seg-{len(segments) + 1}"
        segments.append(
            Segment(
                segment_id=segment_id,
                episode_id=trajectory.episode_id,
                start_step=start_step,
                end_step=end_step,
                reasons=list(dict.fromkeys(current_reasons)) or [BoundaryReason.MAX_EVENTS],
                tool_sequence=[event.action.tool for event in current_events if event.action],
                summary_hint=_summarize_segment(current_events),
                events=list(current_events),
            )
        )

    for index, event in enumerate(trajectory.events):
        current_events.append(event)
        current_reasons.clear()
        next_event = trajectory.events[index + 1] if index + 1 < len(trajectory.events) else None
        if len(current_events) >= max_events_per_segment:
            current_reasons.append(BoundaryReason.MAX_EVENTS)
        if event.success_signal:
            current_reasons.append(BoundaryReason.SUCCESS_SIGNAL)
        if next_event and _tool_switched(event, next_event):
            current_reasons.append(BoundaryReason.TOOL_SWITCH)
        if next_event and _fileset_changed(event, next_event):
            current_reasons.append(BoundaryReason.FILESET_CHANGE)
        if current_reasons or next_event is None:
            flush(event.step_id)
            current_events = []
            start_step = next_event.step_id if next_event else event.step_id

    return segments


def _tool_switched(left: Event, right: Event) -> bool:
    if not left.action or not right.action:
        return False
    return left.action.tool != right.action.tool


def _fileset_changed(left: Event, right: Event) -> bool:
    left_paths = {artifact.path for artifact in left.artifacts if artifact.path}
    right_paths = {artifact.path for artifact in right.artifacts if artifact.path}
    return bool(left_paths and right_paths and left_paths != right_paths)


def _summarize_segment(events: list[Event]) -> str:
    summaries = [event.observation.summary for event in events if event.observation.summary]
    if summaries:
        return summaries[0]
    actions = [event.action.name for event in events if event.action]
    return ", ".join(actions[:3]) if actions else "segment"
