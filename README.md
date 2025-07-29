# 🏭 AI 기반 생산라인 최적화 시뮬레이션

## 📋 프로젝트 개요

### 🎯 프로젝트 목적
이 프로젝트는 **인공지능(AI)을 활용한 최적화 기법을 생산라인 시뮬레이션에 적용**하여, 기존의 경험적 또는 무작위적 의사결정 방식 대비 AI의 효과를 입증하는 것이 목표입니다.

### 🤖 핵심 아이디어
- **문제**: 생산라인에서 각 공정(가공→조립→검사)에 몇 대의 기계를 배치해야 최적의 성과를 낼 수 있을까?
- **AI의 역할**: 실시간으로 생산 상황을 분석하고, 처리량을 최대화하면서 비용과 대기시간을 최소화하는 최적의 기계 배치 전략을 학습
- **검증 방법**: AI 방식 vs 몬테 카를로 방법의 성능을 수치적으로 비교

### 📊 주요 성과 (실제 실행 결과)
- **처리량**: AI가 몬테 카를로 방법 대비 **13.0% 향상** (시간당 생산량: AI 100.9개 vs 몬테 카를로 방법 89.3개)
- **운영비용**: AI가 몬테 카를로 방법 대비 **52.0% 절감** (AI $3,760 vs 몬테 카를로 방법 $7,828)
- **대기시간**: AI가 몬테 카를로 방법 대비 **92.3% 단축** (AI 0.16분 vs 몬테 카를로 방법 2.09분)
- **전체 보상**: AI가 몬테 카를로 방법 대비 **101.8% 향상** (AI 123.7 vs 몬테 카를로 방법 61.3)

## 🛠️ 기술 스택

### 핵심 기술
- **강화학습**: PPO (Proximal Policy Optimization) 알고리즘
- **시뮬레이션**: SimPy (이산사건 시뮬레이션 라이브러리)
- **AI 프레임워크**: Stable-Baselines3
- **데이터 분석**: Pandas, NumPy, Matplotlib, Seaborn

### 개발 환경
- **언어**: Python 3.8+
- **주요 라이브러리**: `gymnasium`, `stable-baselines3`, `simpy`, `matplotlib`, `seaborn`

## 📁 파일 구조 및 역할

### 🔧 핵심 구성 파일

#### 1. `config.py` - 시뮬레이션 설정 관리자
```python
# 주요 설정값 예시
MAX_MACHINES_PER_STATION = 50  # 각 스테이션별 최대 기계 수
SIMULATION_TIME = 60  # 시뮬레이션 시간 (분)
STATION_CONFIG = {
    'machining': {'capacity': 1, 'name': '가공 스테이션'},
    'assembly': {'capacity': 1, 'name': '조립 스테이션'},  
    'inspection': {'capacity': 1, 'name': '검사 스테이션'}
}
```

**역할**: 
- 전체 시스템의 설정값을 중앙집중식으로 관리
- 스테이션별 기계 수, 작업 시간, 부품 투입 간격 등을 설정
- 다양한 시나리오(병목 발생, 고수요 등) 프리셋 제공
- 실험 조건 변경 시 이 파일만 수정하면 전체 시스템에 자동 반영

#### 2. `rl_environment.py` - 강화학습 환경 구현
```python
class ProductionLineEnv(gym.Env):
    def __init__(self):
        # 행동공간: 각 스테이션에 배치할 기계 수 (1~50대)
        self.action_space = spaces.MultiDiscrete([MAX_MACHINES_PER_STATION] * 3)
        
        # 상태공간: [처리량, 평균대기시간, 각 스테이션 가동률]
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0]), 
                                          high=np.array([np.inf, np.inf, 100, 100, 100]))
```

**핵심 동작 로직**:
1. **초기화**: 생산라인 환경을 설정하고 상태공간/행동공간 정의
2. **step() 함수**: AI가 선택한 기계 배치를 받아 시뮬레이션 실행
3. **시뮬레이션 실행**: SimPy를 사용해 실제 생산 과정을 모델링
4. **보상 계산**: 처리량↑, 비용↓, 대기시간↓를 종합한 점수 산출
5. **상태 반환**: 다음 의사결정을 위한 현재 생산라인 상태 제공

#### 3. `simple_agent_v1.py` - AI 에이전트 학습 시스템
```python
class SimpleProductionAgent:
    def train(self, total_timesteps):
        self.model = PPO("MlpPolicy", self.env, 
                        learning_rate=0.0003,
                        n_steps=1024, 
                        batch_size=64)
        self.model.learn(total_timesteps=total_timesteps)
```

**핵심 동작 로직**:
1. **모델 초기화**: PPO 알고리즘으로 정책 네트워크 생성
2. **경험 수집**: 환경과 상호작용하며 (상태, 행동, 보상) 데이터 축적
3. **정책 업데이트**: 수집된 경험을 바탕으로 신경망 가중치 최적화
4. **성능 평가**: 학습된 모델과 몬테 카를로 방법 선택 방식의 성능 비교
5. **모델 저장**: 학습 완료된 AI 모델을 파일로 저장

**PPO 알고리즘 선택 이유**:
- 안정적인 학습 성능
- 연속적/이산적 행동공간 모두 지원
- 산업 현장에서 검증된 신뢰성

#### 4. `training_analysis_v1.py` - 성능 분석 및 시각화 도구
```python
def run_performance_analysis(self, num_episodes=100):
    # AI vs Monte Carlo method 에이전트 성능 데이터 수집
    for agent_type in ['AI', 'Monte Carlo method']:
        for i in range(num_episodes):
            # 각 에피소드별 성능 지표 기록
            agent_data.append({
                'Agent': agent_type,
                'Reward': reward,
                'Throughput': info['throughput'],
                'Cost': info['total_cost'],
                'WaitTime': avg_wait_time
            })
```

**핵심 동작 로직**:
1. **데이터 수집**: AI와 몬테 카를로 방법을 각각 100회씩 실행하여 성능 데이터 축적
2. **통계 분석**: 평균, 중간값, 표준편차 등 기술통계량 계산
3. **비교 분석**: AI 대비 몬테 카를로 방법의 성능 개선률을 백분율로 산출
4. **시각화**: 4개 차트(처리량, 비용, 대기시간, 전략분포)로 결과 표시
5. **보고서 생성**: 터미널과 그래프를 통한 종합 성과 리포트 제공

## 🔄 전체 시스템 동작 흐름

### 1단계: 환경 설정 및 초기화
```
config.py → 시뮬레이션 파라미터 설정
rl_environment.py → 강화학습 환경 구성
```

### 2단계: AI 학습 과정
```
1. 랜덤 기계 배치로 시작
2. 시뮬레이션 실행 → 결과 관찰
3. 보상을 바탕으로 정책 개선
4. 500,000회 반복 학습
```

### 3단계: 성능 검증
```
학습된 AI vs 몬테 카를로 방법 선택 방식
→ 200회 테스트 실행
→ 통계적 유의성 검증
→ 결과 시각화
```

## 🚀 실행 방법

### 1. 의존성 설치
```bash
pip install gymnasium stable-baselines3 simpy matplotlib seaborn pandas numpy
```

### 2. AI 학습 실행
```bash
python simple_agent_v1.py
```
- 학습 시간: 약 30-120분 (컴퓨터 성능에 따라)
- 결과: `my_production_ai.zip` 모델 파일 생성

### 3. 성능 분석 실행
```bash
python training_analysis_v1.py
```

### 4. 설정 변경 (선택사항)
```python
# config.py에서 다양한 실험 조건 설정 가능
MAX_MACHINES_PER_STATION = 30  # 기계 수 제한 변경
apply_scenario('bottleneck_assembly')  # 병목 시나리오 적용
```

## 📈 주요 성과 지표 해석
![스크린샷 2025-06-08 222249](https://github.com/user-attachments/assets/b6e1b7f4-fa34-4210-8f43-7ef81d9a480f)

### 🎯 처리량 (Throughput)
- **의미**: 시간당 완성품 생산 개수
- **AI 성과**: 시간당 생산량 100.9개 (몬테 카를로 방법: 89.3개/시간)
- **비즈니스 임팩트**: 13.0% 생산성 향상 → 매출 직결

### 💰 운영비용 (Cost)
- **의미**: 기계 운영에 필요한 총 비용
- **AI 성과**: AI의 평균 운영 비용은 $3,760으로, 몬테 카를로 방법의 평균 비용($7,828) 대비 52.0% 낮았습니다. 특히 AI는 몬테 카를로 방법이 최대 생산량을 기록했을 때의 최저 비용보다도 25.1% 더 낮은 비용 효율성을 보였습니다.
- **비즈니스 임팩트**: 52.0% 비용 절감 → 수익성 개선

### ⏱️ 대기시간 (Wait Time)
- **의미**: 부품이 각 공정에서 대기하는 평균 시간
- **AI 성과**: 평균 0.16분 (몬테 카를로 방법: 2.09분)
- **비즈니스 임팩트**: 92.3% 대기시간 단축 → 고객 만족도 향상

## 🎓 학습 가치

### 기술적 학습 포인트
1. **강화학습 실전 적용**: 이론을 실제 문제에 적용하는 경험
2. **시뮬레이션 모델링**: 복잡한 시스템을 수학적으로 모델링하는 능력
3. **성능 최적화**: 다목적 최적화 문제 해결 경험
4. **데이터 분석**: 실험 결과를 과학적으로 분석하는 방법론

## 🔮 향후 개선 방향

### 개선 사항
- [ ] 더 복잡한 생산라인 (4개 이상 스테이션) 지원
- [ ] 실시간 수요 변동 반영
- [ ] 기계 고장/유지보수 상황 모델링
- [ ] 다양한 제품 타입 동시 생산

## 📞 연락처 및 추가 정보

이 프로젝트에 대한 추가 문의는 언제든 환영합니다!

---
*이 프로젝트는 AI 기술의 실용적 가치를 입증하고, 제조업 혁신에 기여하고자 하는 목적으로 개발되었습니다.*



# 🏭 AI-Powered Production Line Optimization Simulation

## 📋 Project Overview

### 🎯 Project Purpose
This project aims to **apply AI-powered optimization techniques to production line simulation** and demonstrate the effectiveness of AI compared to traditional empirical or Monte Carlo method decision-making approaches.

### 🤖 Core Concept
- **Problem**: How many machines should be deployed at each production stage (Machining→Assembly→Inspection) to achieve optimal performance?
- **AI's Role**: Analyze production situations in real-time and learn optimal machine allocation strategies that maximize throughput while minimizing costs and wait times
- **Validation Method**: Quantitative performance comparison between AI approach vs. Monte Carlo method

### 📊 Key Performance Results (Actual Execution)
- **Throughput**: AI achieved **13.0% improvement** over Monte Carlo method (AI 100.9 units/hour vs Monte Carlo method 89.3 units/hour)
- **Operating Cost**: AI achieved **52.0% cost reduction** compared to Monte Carlo method (AI $3,760 vs Monte Carlo method $7,828)
- **Wait Time**: AI achieved **92.3% wait time reduction** compared to Monte Carlo method (AI 0.16 min vs Monte Carlo method 2.09 min)
- **Overall Reward**: AI achieved **101.8% improvement** over Monte Carlo method (AI 123.7 vs Monte Carlo method 61.3)

## 🛠️ Technology Stack

### Core Technologies
- **Reinforcement Learning**: PPO (Proximal Policy Optimization) Algorithm
- **Simulation**: SimPy (Discrete Event Simulation Library)
- **AI Framework**: Stable-Baselines3
- **Data Analysis**: Pandas, NumPy, Matplotlib, Seaborn

### Development Environment
- **Language**: Python 3.8+
- **Key Libraries**: `gymnasium`, `stable-baselines3`, `simpy`, `matplotlib`, `seaborn`

## 📁 File Structure and Roles

### 🔧 Core Components

#### 1. `config.py` - Simulation Configuration Manager
```python
# Key configuration examples
MAX_MACHINES_PER_STATION = 50  # Maximum machines per station
SIMULATION_TIME = 60  # Simulation time (minutes)
STATION_CONFIG = {
    'machining': {'capacity': 1, 'name': 'Machining Station'},
    'assembly': {'capacity': 1, 'name': 'Assembly Station'},  
    'inspection': {'capacity': 1, 'name': 'Inspection Station'}
}
```

**Role**: 
- Centralized management of system-wide configuration values
- Configuration of machine counts per station, work times, part arrival intervals, etc.
- Provides various scenario presets (bottlenecks, high demand, etc.)
- Changes to this file automatically reflect across the entire system

#### 2. `rl_environment.py` - Reinforcement Learning Environment Implementation
```python
class ProductionLineEnv(gym.Env):
    def __init__(self):
        # Action space: Number of machines to deploy at each station (1~50 units)
        self.action_space = spaces.MultiDiscrete([MAX_MACHINES_PER_STATION] * 3)
        
        # State space: [throughput, avg_wait_time, utilization_rate_per_station]
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0]), 
                                          high=np.array([np.inf, np.inf, 100, 100, 100]))
```

**Core Operation Logic**:
1. **Initialization**: Set up production line environment and define state/action spaces
2. **step() Function**: Receive AI's machine allocation choice and execute simulation
3. **Simulation Execution**: Model actual production processes using SimPy
4. **Reward Calculation**: Calculate comprehensive score considering throughput↑, cost↓, wait time↓
5. **State Return**: Provide current production line status for next decision-making

#### 3. `simple_agent_v1.py` - AI Agent Training System
```python
class SimpleProductionAgent:
    def train(self, total_timesteps):
        self.model = PPO("MlpPolicy", self.env, 
                        learning_rate=0.0003,
                        n_steps=1024, 
                        batch_size=64)
        self.model.learn(total_timesteps=total_timesteps)
```

**Core Operation Logic**:
1. **Model Initialization**: Create policy network using PPO algorithm
2. **Experience Collection**: Accumulate (state, action, reward) data through environment interaction
3. **Policy Update**: Optimize neural network weights based on collected experience
4. **Performance Evaluation**: Compare performance between trained model and Monte Carlo method selection
5. **Model Saving**: Save trained AI model to file

**PPO Algorithm Selection Rationale**:
- Stable learning performance
- Support for both continuous and discrete action spaces
- Proven reliability in industrial applications

#### 4. `training_analysis_v1.py` - Performance Analysis and Visualization Tool
```python
def run_performance_analysis(self, num_episodes=100):
    # Collect AI vs Monte Carlo method agent performance data
    for agent_type in ['AI', 'Monte Carlo method']:
        for i in range(num_episodes):
            # Record performance metrics for each episode
            agent_data.append({
                'Agent': agent_type,
                'Reward': reward,
                'Throughput': info['throughput'],
                'Cost': info['total_cost'],
                'WaitTime': avg_wait_time
            })
```

**Core Operation Logic**:
1. **Data Collection**: Execute AI and Monte Carlo method 100 times each to accumulate performance data
2. **Statistical Analysis**: Calculate descriptive statistics (mean, median, standard deviation)
3. **Comparative Analysis**: Calculate AI performance improvement rates as percentages
4. **Visualization**: Display results through 4 charts (throughput, cost, wait time, strategy distribution)
5. **Report Generation**: Provide comprehensive performance reports via terminal and graphs

## 🔄 Overall System Workflow

### Stage 1: Environment Setup and Initialization
```
config.py → Set simulation parameters
rl_environment.py → Configure reinforcement learning environment
```

### Stage 2: AI Learning Process
```
1. Start with random machine allocation
2. Execute simulation → Observe results
3. Improve policy based on rewards
4. Repeat learning for 500,000 iterations
```

### Stage 3: Performance Validation
```
Trained AI vs Monte Carlo method selection approach
→ Execute 200 test runs
→ Statistical significance verification
→ Result visualization
```

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install gymnasium stable-baselines3 simpy matplotlib seaborn pandas numpy
```

### 2. Execute AI Training
```bash
python simple_agent_v1.py
```
- Training time: Approximately 30-120 minutes (depending on computer performance)
- Result: Generates `my_production_ai.zip` model file

### 3. Execute Performance Analysis
```bash
python training_analysis_v1.py
```

### 4. Configuration Changes (Optional)
```python
# Various experimental conditions can be set in config.py
MAX_MACHINES_PER_STATION = 30  # Change machine count limit
apply_scenario('bottleneck_assembly')  # Apply bottleneck scenario
```

## 📈 Key Performance Metrics Interpretation
![스크린샷 2025-06-08 222249](https://github.com/user-attachments/assets/dcc27d68-f4e9-4faa-8fbc-540bfe4969ff)

### 🎯 Throughput
- **Meaning**: Number of finished products per hour
- **AI Performance**: 100.9 units/hour (Monte Carlo method: 89.3 units/hour)
- **Business Impact**: 13.0% productivity improvement → Direct revenue impact

### 💰 Operating Cost
- **Meaning**: Total cost required for machine operation
- **AI Performance**: AI's average operating cost was $3,760, which was 52.0% lower than the Monte Carlo method method's average cost ($7,828). Notably, AI demonstrated 25.1% better cost efficiency than the Monte Carlo method method's 'lowest cost' when it achieved 'maximum production'.
- **Business Impact**: 52.0% cost reduction → Profitability improvement

### ⏱️ Wait Time
- **Meaning**: Average time parts wait at each process
- **AI Performance**: Average 0.16 minutes (Monte Carlo method: 2.09 minutes)
- **Business Impact**: 92.3% wait time reduction → Customer satisfaction improvement

## 🎓 Learning Value

### Technical Learning Points
1. **Practical Reinforcement Learning Application**: Experience applying theory to real problems
2. **Simulation Modeling**: Ability to mathematically model complex systems
3. **Performance Optimization**: Experience solving multi-objective optimization problems
4. **Data Analysis**: Methodology for scientifically analyzing experimental results

## 🔮 Future Improvement Directions

### Improvement Items
- [ ] Support for more complex production lines (4+ stations)
- [ ] Real-time demand fluctuation reflection
- [ ] Machine failure/maintenance situation modeling
- [ ] Simultaneous production of various product types

## 📞 Contact and Additional Information

Inquiries about this project are always welcome!

---
*This project was developed to demonstrate the practical value of AI technology and contribute to manufacturing innovation.*
