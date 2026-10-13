"""A compact overview plus focused traffic and event queries."""
from copy import deepcopy
from typing import Any

from common.runtime import parse_phase_id
from methods.llm_tsc.memory import EpisodeMemory
from methods.llm_tsc.state_provider import TrafficStateProvider
from methods.llm_tsc.tools_registry import tool


def normalize_occupancy(values: dict[str, Any]) -> dict[str, float | None]:
    return {
        movement: None if value is None or str(value) in ('-1', '-1.0')
        else round(float(str(value).rstrip('%')), 3)
        for movement, value in values.items()
    }


class TSCToolsAdapter:
    def __init__(self, env: TrafficStateProvider, memory: EpisodeMemory):
        self.env = env
        self.memory = memory
        self.sim_time: float | None = None
        self.cache: dict[str, Any] = {}

    def reset_episode(self) -> None:
        self.memory.reset({
            'layout': self.env.get_intersection_layout(),
            'phases': self.env.get_signal_phase_structure(),
            'available_actions': self.env.get_available_actions(),
            'control_constraints': self.env.get_control_constraints(),
        })
        self.sim_time = None
        self.cache.clear()

    def begin_decision(self, sim_time: float) -> dict:
        if self.sim_time != sim_time:
            self.sim_time = sim_time
            self.cache.clear()
        return self.get_traffic_snapshot()

    def get_traffic_snapshot(self) -> dict:
        if self.sim_time is None:
            raise RuntimeError('Call begin_decision before querying traffic tools')
        if 'snapshot' not in self.cache:
            situation = self.env.get_junction_situation()
            self.cache['situation'] = deepcopy(situation)
            relevant = {key: deepcopy(situation.get(key)) for key in
                        ('rescue_movements', 'movement_access', 'detector_state',
                         'blocked_lanes', 'incident_affected_movements', 'blocked_movements') if key in situation}
            emergency = situation.get('emergency_vehicles', [])
            relevant['emergency_vehicle_count'] = len(emergency)
            relevant['nearest_emergency_distance_m'] = min(
                (vehicle['distance_to_junction_m'] for vehicle in emergency), default=None)
            relevant['accident_count'] = len(situation.get('accident_details', []))
            recent = self.memory.recent(min(3, self.memory.history.maxlen))
            self.cache['snapshot'] = {
                'schema_version': 4, 'sim_time': self.sim_time,
                'current_phase': self.env.get_current_phase(),
                'situation': relevant,
                'occupancy': normalize_occupancy(self.env.get_current_occupancy()),
                'previous_occupancy': deepcopy(recent[-1]['occupancy']) if recent else
                                      normalize_occupancy(self.env.get_previous_occupancy()),
                'previous_sim_time': recent[-1]['sim_time'] if recent else None,
                'recent_actions': [{'sim_time': item['sim_time'], 'phase_id': item['phase_id']}
                                   for item in recent],
                'occupancy_unit': 'percent',
            }
            self.cache['snapshot']['expert_advice'] = {
                'method': 'MP', **deepcopy(self._get_phase_recommendation()),
            }
        return deepcopy(self.cache['snapshot'])

    @tool(description='No arguments. Read up to five recent observations and executed phases only when a longer demand trend is needed. Current and previous occupancy are already in the supplied snapshot.')
    def get_traffic_history(self) -> dict:
        snapshot = self.get_traffic_snapshot()
        recent = self.memory.recent(self.memory.history.maxlen)
        return {'sim_time': snapshot['sim_time'], 'occupancy_unit': 'percent',
                'recent_decisions': [{'sim_time': item['sim_time'], 'phase_id': item['phase_id'],
                                      'occupancy': item['occupancy']} for item in recent]}

    def _get_phase_recommendation(self) -> dict:
        if 'recommendation' not in self.cache:
            recommendation = deepcopy(self.env.get_traditional_decision())
            phase_id = parse_phase_id(recommendation.get('recommended_phase'))
            recommendation['recommended_movements'] = next(
                (list(info['movements']) for info in self.memory.static_context['phases'].values()
                 if info['phase_id'] == phase_id), [])
            self.cache['recommendation'] = recommendation
        return self.cache['recommendation']

    @tool(description='No arguments. Read the same MP expert advice already displayed in snapshot.expert_advice: phase scores, recommended phase and served movements. No extra call is needed to receive this advice. The LLM chooses whether to adopt or adjust it using live traffic and special events.')
    def get_phase_recommendation(self) -> dict:
        snapshot = self.get_traffic_snapshot()
        return {'sim_time': snapshot['sim_time'],
                'recommendation': deepcopy(self._get_phase_recommendation())}

    @tool(description='Map current special-vehicle routes through this junction to movement, serving phases and eligible phases after lane-incident checks. permissive_phases require yielding. Includes priority, distance and waiting for comparing vehicles. Use for special-vehicle priority decisions.')
    def get_emergency_vehicles(self) -> dict:
        snapshot = self.get_traffic_snapshot()
        return {'sim_time': snapshot['sim_time'],
                'vehicles': deepcopy(self.cache['situation'].get('emergency_vehicles', []))}

    @tool(description='Inspect active incidents by lane: affected movements, blocked lanes, remaining usable lanes and whether a movement is fully blocked. Use when incident impact changes the phase choice.')
    def get_incident_details(self) -> dict:
        snapshot = self.get_traffic_snapshot()
        situation = self.cache['situation']
        return {'sim_time': snapshot['sim_time'],
                'accidents': deepcopy(situation.get('accident_details', [])),
                'movement_impacts': deepcopy(situation.get('movement_impacts', {}))}

    @tool(description='Inspect detector failures and affected missing occupancy readings. A detector failure is missing information, not zero demand or a road closure.')
    def get_sensor_status(self) -> dict:
        snapshot = self.get_traffic_snapshot()
        states = self.cache['situation'].get('detector_state', {})
        failed = [movement for movement, state in states.items() if state != 'Work']
        return {'sim_time': snapshot['sim_time'], 'detector_state': deepcopy(states),
                'failed_movements': failed,
                'previous_sim_time': snapshot['previous_sim_time'],
                'current_readings': {movement: snapshot['occupancy'].get(movement) for movement in failed},
                'previous_readings': {movement: snapshot['previous_occupancy'].get(movement) for movement in failed}}

    @tool(description='Inspect one movement when aggregate occupancy is insufficient: per-lane vehicle and stopped counts, speed, incident blockage and remaining lanes. Readings remain null when its detector is faulty.')
    def inspect_movement(self, movement_id: str) -> dict:
        snapshot = self.get_traffic_snapshot()
        key = f'movement:{movement_id}'
        if key not in self.cache:
            self.cache[key] = deepcopy(self.env.get_movement_details(movement_id))
        return {'sim_time': snapshot['sim_time'], **deepcopy(self.cache[key])}
