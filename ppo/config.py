# 🏭 생산라인 시뮬레이션 설정 파일

# ==================================================
# ⚙️ 핵심 제어 설정
# ==================================================
# 각 스테이션에 할당할 수 있는 기계의 최대 대수
# 이 값을 변경하면 환경, 에이전트, 분석 코드가 모두 자동으로 적응합니다.
MAX_MACHINES_PER_STATION = 50

# ==================================================
# 📋 기본 시뮬레이션 설정
# ==================================================
SIMULATION_TIME = 60  # 시뮬레이션 실행 시간 (분)
RANDOM_SEED = 42      # 재현 가능한 결과를 위한 시드값

# ==================================================
# 🏗️ 스테이션 설정 (기계 대수)
# 이 값은 시뮬레이션 시작 시 동적으로 변경됩니다.
# ==================================================
STATION_CONFIG = {
    'machining': {
        'capacity': 1,           # 가공 기계 대수 (초기값)
        'name': '가공 스테이션'
    },
    'assembly': {
        'capacity': 1,           # 조립 기계 대수 (초기값)
        'name': '조립 스테이션'
    },
    'inspection': {
        'capacity': 1,           # 검사 기계 대수 (초기값)
        'name': '검사 스테이션'
    }
}

# ==================================================
# ⏱️ 작업 시간 설정 (분 단위)
# ==================================================
WORK_TIME = {
    'machining': {
        'min': 2.0,             # 가공 최소 시간
        'max': 4.0              # 가공 최대 시간
    },
    'assembly': {
        'min': 3.0,             # 조립 최소 시간
        'max': 5.0              # 조립 최대 시간
    },
    'inspection': {
        'min': 1.0,             # 검사 최소 시간
        'max': 2.0              # 검사 최대 시간
    }
}

# ==================================================
# 📦 부품 투입 설정
# ==================================================
PART_ARRIVAL = {
    'min_interval': 1.0,        # 부품 투입 최소 간격 (분)
    'max_interval': 3.0,        # 부품 투입 최대 간격 (분)
    'max_parts': 100            # 최대 생산 부품 수 (0 = 무제한)
}

# ==================================================
# 🔍 품질 설정
# ==================================================
QUALITY = {
    'pass_rate': 0.9,           # 합격률 (90%)
    'rework_enabled': False     # 재작업 기능 사용 여부
}

# ==================================================
# 📊 시나리오 프리셋
# ==================================================
SCENARIOS = {
    'default': {
        'name': '기본 설정',
        'description': '균형잡힌 일반적인 생산라인'
    },

    'bottleneck_assembly': {
        'name': '조립 병목 시나리오',
        'description': '조립이 느려서 병목이 발생하는 상황',
        'station_config': {
            'assembly': {'capacity': 1}
        },
        'work_time': {
            'assembly': {'min': 5.0, 'max': 8.0}  # 조립이 더 오래 걸림
        }
    },

    'high_demand': {
        'name': '고수요 시나리오',
        'description': '부품 투입이 매우 빈번한 상황',
        'part_arrival': {
            'min_interval': 0.5,    # 부품이 더 자주 들어옴
            'max_interval': 1.5
        }
    },

    'quality_issue': {
        'name': '품질 문제 시나리오',
        'description': '불량률이 높은 상황',
        'quality': {
            'pass_rate': 0.7        # 합격률 70%로 낮춤
        }
    }
}

# ==================================================
# 🛠️ 설정 적용 함수
# ==================================================
def apply_scenario(scenario_name):
    """특정 시나리오의 설정을 적용"""
    global STATION_CONFIG, WORK_TIME, PART_ARRIVAL, QUALITY

    if scenario_name not in SCENARIOS:
        print(f"❌ 시나리오 '{scenario_name}'를 찾을 수 없습니다.")
        return False

    scenario = SCENARIOS[scenario_name]
    print(f"🎯 시나리오 적용: {scenario['name']}")
    print(f"📝 설명: {scenario['description']}")

    # 각 설정 카테고리별로 업데이트
    if 'station_config' in scenario:
        for station, config in scenario['station_config'].items():
            STATION_CONFIG[station].update(config)

    if 'work_time' in scenario:
        for station, time_config in scenario['work_time'].items():
            WORK_TIME[station].update(time_config)

    if 'part_arrival' in scenario:
        PART_ARRIVAL.update(scenario['part_arrival'])

    if 'quality' in scenario:
        QUALITY.update(scenario['quality'])

    print("✅ 시나리오 설정이 적용되었습니다.\n")
    return True

def print_current_config():
    """현재 설정 상태를 출력"""
    print("📋 현재 시뮬레이션 설정:")
    print(f"   시뮬레이션 시간: {SIMULATION_TIME}분")
    print(f"   스테이션별 최대 기계 수: {MAX_MACHINES_PER_STATION}대")
    print(f"   부품 투입 간격: {PART_ARRIVAL['min_interval']}-{PART_ARRIVAL['max_interval']}분")
    print(f"   품질 합격률: {QUALITY['pass_rate']*100}%")

    print("\n🏗️ 현재 스테이션 상태:")
    for station, config in STATION_CONFIG.items():
        capacity = config['capacity']
        work_time = WORK_TIME[station]
        print(f"   {config['name']}: {capacity}대, {work_time['min']}-{work_time['max']}분")

# ==================================================
# 🧪 테스트 코드
# ==================================================
if __name__ == "__main__":
    print("=== 설정 파일 테스트 ===")

    # 기본 설정 출력
    print_current_config()

    print("\n" + "-"*40)

    # 시나리오 적용 테스트
    apply_scenario('bottleneck_assembly')
    print_current_config()
