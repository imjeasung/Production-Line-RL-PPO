import gymnasium as gym
from gymnasium import spaces
import numpy as np
import simpy
import random
# config.py에서 동적 설정을 가져옵니다.
from config import (
    STATION_CONFIG, WORK_TIME, PART_ARRIVAL,
    MAX_MACHINES_PER_STATION, # <--- 핵심 변경: 최대 기계 수 설정 가져오기
    apply_scenario
)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

class ProductionLineEnv(gym.Env):
    """
    생산라인 최적화를 위한 동적 강화학습 환경.
    config.py의 MAX_MACHINES_PER_STATION 값에 따라 행동 공간이 자동으로 조절됩니다.
    """

    def __init__(self):
        super().__init__()

        # === 핵심 변경: 행동 공간을 동적으로 설정 ===
        # 이전: spaces.Box(low=np.array([1, 1, 1]), high=np.array([3, 3, 3]))
        # 변경: spaces.MultiDiscrete를 사용하여 1부터 MAX_MACHINES_PER_STATION까지의
        #       이산적인 기계 대수를 선택하도록 합니다.
        #       [MAX, MAX, MAX]는 각 스테이션이 0 ~ MAX-1 까지의 값을 가짐을 의미합니다.
        #       step 함수에서 이 값에 +1을 하여 실제 기계 대수(1 ~ MAX)로 변환합니다.
        self.action_space = spaces.MultiDiscrete([MAX_MACHINES_PER_STATION] * 3)

        # 상태 공간: [처리량, 평균대기시간, 각 스테이션 가동률(3개)] = 총 5개
        # 최대값은 일반적인 상황을 가정한 것이며, 시뮬레이션 결과가 이 값을 넘을 수도 있습니다.
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([np.inf, np.inf, 100, 100, 100]), # 처리량과 대기시간은 무한대로 설정
            dtype=np.float32
        )

        # 시뮬레이션 설정
        self.simulation_time = 30  # 빠른 학습을 위해 에피소드당 시뮬레이션 시간 단축
        self.max_steps = 10        # 에피소드당 최대 스텝 수
        self.current_step = 0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 시나리오를 기본값으로 초기화
        apply_scenario('default')
        # 상태 초기화
        self.current_state = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.current_step = 0
        info = {'message': 'Environment reset'}
        return self.current_state, info

    def step(self, action):
        # === 핵심 변경: Action 해석 부분을 0-based 인덱스에 맞게 수정 ===
        # action은 이제 [0, 0, 0] ~ [MAX-1, MAX-1, MAX-1] 범위의 값을 가집니다.
        # 여기에 1을 더해 실제 기계 수 (1 ~ MAX)를 계산합니다.
        machining_machines = action[0] + 1
        assembly_machines = action[1] + 1
        inspection_machines = action[2] + 1

        # 시뮬레이션 환경에 기계 대수 설정 적용
        STATION_CONFIG['machining']['capacity'] = machining_machines
        STATION_CONFIG['assembly']['capacity'] = assembly_machines
        STATION_CONFIG['inspection']['capacity'] = inspection_machines

        # 시뮬레이션 실행 및 결과 반환
        result = self._run_simulation()

        # 관측 상태 업데이트
        self.current_state = np.array([
            result['throughput'],
            result['avg_wait_time'],
            result['utilization']['machining'],
            result['utilization']['assembly'],
            result['utilization']['inspection']
        ], dtype=np.float32)

        # 보상 계산
        reward = self._calculate_reward(result, action)

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False # 시간 초과 외 다른 종료 조건은 없음

        info = {
            'throughput': result['throughput'],
            'total_cost': result['total_cost'],
            'machines_used': machining_machines + assembly_machines + inspection_machines
        }

        return self.current_state, reward, terminated, truncated, info

    def _run_simulation(self):
        """설정된 조건으로 SimPy 시뮬레이션 실행"""
        production_data = []
        station_usage = {station: 0 for station in STATION_CONFIG.keys()}

        def part_process(env, part_name, stations):
            """부품 생산 공정"""
            arrival_time = env.now
            wait_times = []

            for station_name, station_info in STATION_CONFIG.items():
                station_resource = stations[station_name]
                work_time_config = WORK_TIME[station_name]

                wait_start = env.now
                with station_resource.request() as request:
                    yield request
                    wait_time = env.now - wait_start
                    wait_times.append(wait_time)

                    work_time = random.uniform(work_time_config['min'], work_time_config['max'])
                    yield env.timeout(work_time)
                    station_usage[station_name] += work_time

            production_data.append({
                'total_time': env.now - arrival_time,
                'wait_times': wait_times
            })

        def part_generator(env, stations):
            """부품 생성기"""
            part_num = 1
            while True:
                # 부품 도착 간격 설정
                yield env.timeout(random.uniform(PART_ARRIVAL['min_interval'], PART_ARRIVAL['max_interval']))
                env.process(part_process(env, f"Part-{part_num}", stations))
                part_num += 1

        env = simpy.Environment()
        stations = {name: simpy.Resource(env, capacity=config['capacity']) for name, config in STATION_CONFIG.items()}
        env.process(part_generator(env, stations))
        env.run(until=self.simulation_time)

        if not production_data:
            return {
                'throughput': 0,
                'avg_wait_time': self.simulation_time, # 대기시간을 최대로 설정하여 페널티 부여
                'utilization': {station: 0 for station in STATION_CONFIG.keys()},
                'total_cost': sum(STATION_CONFIG[s]['capacity'] * c for s, c in {'machining': 100, 'assembly': 120, 'inspection': 80}.items())
            }

        throughput = len(production_data) / self.simulation_time * 60  # 시간당 처리량
        all_wait_times = [wt for data in production_data for wt in data['wait_times']]
        avg_wait_time = np.mean(all_wait_times) if all_wait_times else 0

        utilization = {}
        for station, usage in station_usage.items():
            capacity = STATION_CONFIG[station]['capacity']
            if capacity > 0:
                utilization[station] = (usage / (capacity * self.simulation_time)) * 100
            else:
                utilization[station] = 0

        cost_per_machine = {'machining': 100, 'assembly': 120, 'inspection': 80}
        total_cost = sum(STATION_CONFIG[s]['capacity'] * cost_per_machine[s] for s in STATION_CONFIG.keys())

        return {
            'throughput': throughput,
            'avg_wait_time': avg_wait_time,
            'utilization': utilization,
            'total_cost': total_cost
        }

    def _calculate_reward(self, result, action):
        """보상 함수: 처리량 최대화, 비용 및 대기시간 최소화"""
        throughput = result['throughput']
        total_cost = result['total_cost']
        avg_wait_time = result['avg_wait_time']

        # 보상 설계: (처리량 가중치 * 처리량) - (비용 가중치 * 비용) - (대기시간 가중치 * 대기시간)
        reward = (throughput * 1.5) - (total_cost * 0.01) - (avg_wait_time * 0.5)

        # 처리량이 일정 수준 이상일 때 보너스
        if throughput > (20 * (self.simulation_time / 60)): # 시간 비율에 맞춘 목표 처리량
             reward += 10

        # === 핵심 변경: 과도한 기계 사용에 대한 페널티를 동적으로 설정 ===
        # 총 기계 대수 계산 (action은 0-based이므로 +1씩 해줌)
        total_machines = sum(action) + len(action)
        # 최대 가능 기계 대수의 60%를 초과하면 페널티 부여
        penalty_threshold = (MAX_MACHINES_PER_STATION * len(action)) * 0.6
        if total_machines > penalty_threshold:
            reward -= (total_machines - penalty_threshold) * 0.5 # 초과분에 비례하여 페널티

        return reward

# 테스트 코드
if __name__ == "__main__":
    print("🤖 강화학습 환경 동적 설정 테스트")
    print(f"🔩 스테이션별 최대 기계 수: {MAX_MACHINES_PER_STATION}대")

    env = ProductionLineEnv()
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")

    state, info = env.reset()
    print(f"\n초기 상태: {state}")
    print(f"초기 정보: {info}")

    for i in range(3):
        action = env.action_space.sample()  # 랜덤 액션
        machines = action + 1 # 실제 기계 대수
        print(f"\n=== 테스트 {i+1} ===")
        print(f"▶️  실행 액션 (0-based): {action} -> 기계 배치: 가공({machines[0]}), 조립({machines[1]}), 검사({machines[2]})")

        state, reward, terminated, truncated, info = env.step(action)

        print(f"   상태 (관측): {np.round(state, 2)}")
        print(f"   보상: {reward:.2f}")
        print(f"   정보: 처리량({info['throughput']:.1f}), 비용(${info['total_cost']}), 총기계({info['machines_used']})")
        if terminated or truncated:
            print("   에피소드 종료됨.")
            env.reset()