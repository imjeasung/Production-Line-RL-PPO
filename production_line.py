import simpy
import random

def part_process(env, part_name, machining_station, assembly_station, inspection_station):
    """부품이 3단계 생산라인을 거치는 전체 과정"""
    arrival_time = env.now
    print(f"{env.now:.1f}: {part_name} 생산라인에 투입")
    
    # 1단계: 가공 스테이션
    with machining_station.request() as request:
        yield request
        print(f"{env.now:.1f}: {part_name} 가공 시작")
        yield env.timeout(random.uniform(2, 4))  # 가공시간 2-4분
        print(f"{env.now:.1f}: {part_name} 가공 완료")
    
    # 2단계: 조립 스테이션
    with assembly_station.request() as request:
        yield request
        print(f"{env.now:.1f}: {part_name} 조립 시작")
        yield env.timeout(random.uniform(3, 5))  # 조립시간 3-5분
        print(f"{env.now:.1f}: {part_name} 조립 완료")
    
    # 3단계: 검사 스테이션
    with inspection_station.request() as request:
        yield request
        print(f"{env.now:.1f}: {part_name} 검사 시작")
        yield env.timeout(random.uniform(1, 2))  # 검사시간 1-2분
        
        # 품질 검사 결과 (90% 합격률)
        is_pass = random.random() < 0.9
        if is_pass:
            print(f"{env.now:.1f}: {part_name} 검사 합격 ✓")
        else:
            print(f"{env.now:.1f}: {part_name} 검사 불합격 ✗")
    
    total_time = env.now - arrival_time
    print(f"{env.now:.1f}: {part_name} 생산 완료 (총 소요시간: {total_time:.1f}분)\n")

def part_generator(env, machining_station, assembly_station, inspection_station):
    """부품을 지속적으로 생성하여 생산라인에 투입"""
    part_number = 1
    
    while True:
        # 새 부품 도착 간격 (1-3분)
        yield env.timeout(random.uniform(1, 3))
        
        # 부품을 생산라인으로 보내기
        part_name = f"부품{part_number:02d}"
        env.process(part_process(env, part_name, machining_station, assembly_station, inspection_station))
        part_number += 1

# 시뮬레이션 환경 생성
env = simpy.Environment()

# 각 스테이션 생성 (각각 1대씩)
machining_station = simpy.Resource(env, capacity=1)    # 가공 스테이션
assembly_station = simpy.Resource(env, capacity=1)     # 조립 스테이션  
inspection_station = simpy.Resource(env, capacity=1)   # 검사 스테이션

# 부품 생성기 시작
env.process(part_generator(env, machining_station, assembly_station, inspection_station))

# 시뮬레이션 실행 (30분간)
print("=== 3단계 생산라인 시뮬레이션 시작 ===")
env.run(until=30)
print("=== 시뮬레이션 종료 ===")