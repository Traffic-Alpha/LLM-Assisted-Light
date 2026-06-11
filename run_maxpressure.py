'''
@Author: WANG Maonan
@Date: 2026-06-11 18:09:09
@Description: Run the max-pressure baseline on the same scenarios/events.
--> python run_maxpressure.py --scenario 4way --event-config scenarios/4way/events/accident_set1.yaml --gui
@LastEditTime: 2026-06-12 00:51:46
'''
import argparse

from llm_tsc.logger import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from llm_tsc.maxpressure import MaxPressureController
from traffic_env.base_tsc_wrapper import BaseTSCEnvWrapper
from traffic_env.event_wrapper import create_event_wrapper
from traffic_env.tsc_env import TrafficSignalEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Max-pressure TSC baseline")
    parser.add_argument("--scenario", default="4way", help="Scenario name under scenarios/")
    parser.add_argument("--phase-num", type=int, default=4, help="Number of TLS phases")
    parser.add_argument("--event-config", default=None, help="Full event YAML path")
    parser.add_argument("--tls-id", default="J1", help="Traffic light id")
    parser.add_argument("--num-seconds", type=int, default=500, help="Simulation horizon")
    parser.add_argument("--gui", action="store_true", help="Run SUMO with GUI")
    args = parser.parse_args()

    path_convert = get_abs_path(__file__)
    set_logger(path_convert("./"))
    event_config = args.event_config or path_convert(
        f"./scenarios/{args.scenario}/events/default.yaml"
    ) # 默认的 event 配置文件路径为 scenarios/{scenario}/events/default.yaml

    env = TrafficSignalEnv(
        sumo_cfg=path_convert(f"./scenarios/{args.scenario}/env/vehicle.sumocfg"),
        net_file=path_convert(f"./scenarios/{args.scenario}/env/{args.scenario}.net.xml"),
        trip_info=path_convert(f"./{args.scenario}_maxpressure.tripinfo.xml"),
        num_seconds=args.num_seconds,
        tls_id=args.tls_id,
        tls_action_type="choose_next_phase",
        use_gui=args.gui,
    )
    wrapped = BaseTSCEnvWrapper(env=env, tls_id=args.tls_id, phase_num=args.phase_num)
    wrapped = create_event_wrapper(wrapped, args.scenario, event_config)
    controller = MaxPressureController(args.phase_num)

    done = False
    sim_time = 0
    phase_id = 0
    last_info = {}
    wrapped.reset()

    while not done and sim_time < args.num_seconds:
        phase_id = controller.act(last_info, fallback_phase=phase_id)
        _, _, _, done, info = wrapped.step(phase_id)
        last_info = info
        sim_time = info["step_time"]

    wrapped.close()
    logger.info("Finished max-pressure baseline at {}s", sim_time)


if __name__ == "__main__":
    main()
