import simpy
import random

# 전역 데이터 수집 변수
production_data = []
station_usage = {'machining': 0, 'assembly': 0, 'inspection': 0}

def part_process(env, part_name, machining_station, assembly_station, inspection_station):
    """부품 생산 과정 + 데이터 수집"""
    # 데이터 기록용 딕셔너리
    data = {
        'part_name': part_name,
        'arrival_time': env.now,
        'machining_start': 0,
        'machining_end': 0,
        'assembly_start': 0, 
        'assembly_end': 0,
        'inspection_start': 0,
        'inspection_end': 0,
        'total_time': 0,
        'is_pass': False
    }
    
    print(f"{env.now:.1f}: {part_name} 생산라인 투입")
    
    # 1단계: 가공
    with machining_station.request() as request:
        yield request
        data['machining_start'] = env.now
        print(f"{env.now:.1f}: {part_name} 가공 시작")
        
        work_time = random.uniform(2, 4)
        yield env.timeout(work_time)
        station_usage['machining'] += work_time
        
        data['machining_end'] = env.now
        print(f"{env.now:.1f}: {part_name} 가공 완료")
    
    # 2단계: 조립
    with assembly_station.request() as request:
        yield request
        data['assembly_start'] = env.now
        print(f"{env.now:.1f}: {part_name} 조립 시작")
        
        work_time = random.uniform(3, 5)
        yield env.timeout(work_time)
        station_usage['assembly'] += work_time
        
        data['assembly_end'] = env.now
        print(f"{env.now:.1f}: {part_name} 조립 완료")
    
    # 3단계: 검사
    with inspection_station.request() as request:
        yield request
        data['inspection_start'] = env.now
        print(f"{env.now:.1f}: {part_name} 검사 시작")
        
        work_time = random.uniform(1, 2)
        yield env.timeout(work_time)
        station_usage['inspection'] += work_time
        
        data['inspection_end'] = env.now
        data['is_pass'] = random.random() < 0.9
        
        if data['is_pass']:
            print(f"{env.now:.1f}: {part_name} 검사 합격 ✓")
        else:
            print(f"{env.now:.1f}: {part_name} 검사 불합격 ✗")
    
    # 총 시간 계산
    data['total_time'] = env.now - data['arrival_time']
    production_data.append(data)
    print(f"{env.now:.1f}: {part_name} 완료 (총 {data['total_time']:.1f}분)\n")

def part_generator(env, machining_station, assembly_station, inspection_station):
    """부품 생성기"""
    part_number = 1
    while True:
        yield env.timeout(random.uniform(1, 3))
        part_name = f"부품{part_number:02d}"
        env.process(part_process(env, part_name, machining_station, assembly_station, inspection_station))
        part_number += 1

def analyze_production_data(simulation_time):
    """생산 데이터 분석 및 리포트 출력"""
    if not production_data:
        print("수집된 데이터가 없습니다.")
        return
    
    print("\n" + "="*50)
    print("📊 생산라인 성과 분석 리포트")
    print("="*50)
    
    # 기본 통계
    completed_parts = len(production_data)
    passed_parts = sum(1 for data in production_data if data['is_pass'])
    
    print(f"🏭 총 생산량: {completed_parts}개")
    print(f"✅ 합격품: {passed_parts}개")
    print(f"❌ 불합격품: {completed_parts - passed_parts}개")
    print(f"📈 합격률: {passed_parts/completed_parts*100:.1f}%")
    print(f"⚡ 처리량: {completed_parts/simulation_time*60:.1f}개/시간")
    
    # 평균 시간 분석
    avg_total = sum(data['total_time'] for data in production_data) / completed_parts
    print(f"⏱️  평균 총 소요시간: {avg_total:.1f}분")
    
    # 각 스테이션 가동률 계산
    print(f"\n🔧 스테이션 가동률:")
    for station, usage_time in station_usage.items():
        utilization = (usage_time / simulation_time) * 100
        print(f"   {station:12}: {utilization:.1f}%")
    
    # 병목 구간 찾기
    max_usage = max(station_usage.values())
    bottleneck = [k for k, v in station_usage.items() if v == max_usage][0]
    print(f"🚨 병목 구간: {bottleneck} 스테이션")

# 시뮬레이션 실행
env = simpy.Environment()

# 스테이션 생성
machining_station = simpy.Resource(env, capacity=1)
assembly_station = simpy.Resource(env, capacity=1)
inspection_station = simpy.Resource(env, capacity=1)

# 부품 생성기 시작
env.process(part_generator(env, machining_station, assembly_station, inspection_station))

# 시뮬레이션 실행
simulation_time = 30
print("=== 데이터 수집 시뮬레이션 시작 ===")
env.run(until=simulation_time)

# 결과 분석
analyze_production_data(simulation_time)