import simpy
import random

def machine_process(env, name, machine, process_time):
    """부품이 기계에서 가공되는 과정"""
    print(f"{env.now:.1f}: {name} 부품이 기계에 도착했습니다")
    
    # 기계 사용 요청 (한 번에 하나만 사용 가능)
    with machine.request() as request:
        yield request
        print(f"{env.now:.1f}: {name} 부품 가공을 시작합니다")
        
        # 가공 시간만큼 대기
        yield env.timeout(process_time)
        print(f"{env.now:.1f}: {name} 부품 가공이 완료되었습니다")

def part_generator(env, machine):
    """부품을 계속 생성하는 함수"""
    part_number = 1
    
    while True:
        # 새 부품 도착 간격 (1-3분 랜덤)
        yield env.timeout(random.uniform(1, 3))
        
        # 가공 시간 (2-4분 랜덤)
        process_time = random.uniform(2, 4)
        
        # 부품을 기계로 보내기
        env.process(machine_process(env, f"부품{part_number}", machine, process_time))
        part_number += 1

# 시뮬레이션 환경 생성
env = simpy.Environment()

# 기계 생성 (한 번에 1개 부품만 처리 가능)
machine = simpy.Resource(env, capacity=1)

# 부품 생성기 시작
env.process(part_generator(env, machine))

# 시뮬레이션 실행 (10분간)
print("=== 생산라인 시뮬레이션 시작 ===")
env.run(until=10)
print("=== 시뮬레이션 종료 ===")