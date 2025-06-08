from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import numpy as np
# 수정된 동적 환경을 임포트합니다.
from rl_environment import ProductionLineEnv
# config에서 최대 기계 대수 설정을 가져옵니다.
from config import MAX_MACHINES_PER_STATION

class SimpleProductionAgent:
    """
    동적 생산라인 환경에 맞춰 학습하는 AI 에이전트.
    rl_environment.py의 변경사항(MultiDiscrete Action Space)을 자동으로 감지하고 작동합니다.
    """

    def __init__(self):
        self.env = ProductionLineEnv()
        self.model = None
        self.is_trained = False

    def train(self, total_timesteps):
        """AI 에이전트 학습"""
        print("🔍 환경 검증 중...")

        try:
            # check_env가 동적으로 설정된 환경(MultiDiscrete 포함)을 검증합니다.
            check_env(self.env)
            print("✅ 환경 검증 완료!")
        except Exception as e:
            print(f"❌ 환경 오류: {e}")
            return False

        print(f"🧠 AI 학습 시작 (총 {total_timesteps} 스텝)")

        # PPO 알고리즘은 'MlpPolicy'를 통해 MultiDiscrete 행동 공간을 지원합니다.
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=1024,      # 더 많은 데이터를 수집 후 업데이트하여 안정성 향상
            batch_size=64,
            gamma=0.99,        # 미래 보상에 대한 할인율
            ent_coef=0.01,     # 탐험을 장려하기 위한 엔트로피 계수
            clip_range=0.2,    # PPO의 클리핑 범위
            tensorboard_log="./ppo_production_tensorboard/" # 학습 과정을 텐서보드에 기록
        )

        # 학습 실행
        self.model.learn(total_timesteps=total_timesteps)
        self.is_trained = True

        print("🎉 학습 완료!")
        return True

    def test_agent(self, num_tests=5):
        """학습된 에이전트 성능 테스트"""
        if not self.is_trained:
            print("❌ 먼저 학습을 완료해주세요!")
            return None

        print(f"\n🧪 학습된 AI 성능 테스트 ({num_tests}회)")
        print("="*50)

        results = []
        total_rewards = []
        total_throughputs = []
        total_costs = []

        for test_num in range(num_tests):
            obs, _ = self.env.reset()
            action, _ = self.model.predict(obs, deterministic=True)
            # action 값에 1을 더해 실제 기계 대수를 계산합니다 (0-based -> 1-based)
            machines = action + 1

            next_obs, reward, terminated, truncated, info = self.env.step(action)

            results.append({
                'test_num': test_num + 1,
                'action': machines, # 실제 기계 대수로 저장
                'reward': reward,
                'throughput': info['throughput'],
                'cost': info['total_cost']
            })
            total_rewards.append(reward)
            total_throughputs.append(info['throughput'])
            total_costs.append(info['total_cost'])

            print(f"테스트 {test_num + 1}:")
            print(f"  기계 배치: 가공({machines[0]}), 조립({machines[1]}), 검사({machines[2]})")
            print(f"  처리량: {info['throughput']:.1f}개/시간, 비용: ${info['total_cost']}, 보상: {reward:.2f}")

        print("\n📊 AI 성능 요약:")
        print(f"  평균 처리량: {np.mean(total_throughputs):.1f} 개/시간")
        print(f"  평균 비용: ${np.mean(total_costs):.0f}")
        print(f"  평균 보상: {np.mean(total_rewards):.2f}")

        return results

    def compare_with_random(self, num_trials=5):
        """랜덤 선택과 AI 성능 비교"""
        if not self.is_trained:
            print("❌ 먼저 학습을 완료해주세요!")
            return

        print(f"\n🥊 AI vs 랜덤 성능 비교 ({num_trials}회 평균)")
        print("="*50)

        # 1. AI 성능 측정
        ai_rewards, ai_throughputs = [], []
        for _ in range(num_trials):
            obs, _ = self.env.reset()
            ai_action, _ = self.model.predict(obs, deterministic=True)
            _, reward, _, _, info = self.env.step(ai_action)
            ai_rewards.append(reward)
            ai_throughputs.append(info['throughput'])
        
        avg_ai_reward = np.mean(ai_rewards)
        avg_ai_throughput = np.mean(ai_throughputs)

        # 2. 랜덤 성능 측정
        random_rewards, random_throughputs = [], []
        for _ in range(num_trials):
            obs, _ = self.env.reset()
            random_action = self.env.action_space.sample()
            _, reward, _, _, info = self.env.step(random_action)
            random_rewards.append(reward)
            random_throughputs.append(info['throughput'])

        avg_random_reward = np.mean(random_rewards)
        avg_random_throughput = np.mean(random_throughputs)

        # 결과 출력
        obs, _ = self.env.reset()
        ai_action, _ = self.model.predict(obs, deterministic=True)
        machines = ai_action + 1

        print(f"🤖 AI 에이전트 (평균):")
        print(f"  - 대표 전략: 가공({machines[0]}), 조립({machines[1]}), 검사({machines[2]})")
        print(f"  - 평균 처리량: {avg_ai_throughput:.1f} 개/시간")
        print(f"  - 평균 보상: {avg_ai_reward:.2f}")
        print()
        print(f"🎲 랜덤 선택 (평균):")
        print(f"  - 평균 처리량: {avg_random_throughput:.1f} 개/시간")
        print(f"  - 평균 보상: {avg_random_reward:.2f}")
        print()

        # 성능 향상도 계산
        throughput_improvement = ((avg_ai_throughput - avg_random_throughput) / (avg_random_throughput + 1e-6)) * 100
        reward_improvement = ((avg_ai_reward - avg_random_reward) / (abs(avg_random_reward) + 1e-6)) * 100

        print(f"📈 AI 성능 향상:")
        print(f"  처리량 향상: {throughput_improvement:+.1f}%")
        print(f"  보상 향상: {reward_improvement:+.1f}%")

        if avg_ai_reward > avg_random_reward:
            print("🏆 AI가 랜덤보다 우수한 성능을 보입니다!")
        else:
            print("🤔 AI가 더 학습이 필요해 보입니다.")

    def save_model(self, filename="production_agent"):
        """학습된 모델 저장"""
        if not self.is_trained:
            print("❌ 저장할 모델이 없습니다!")
            return False
        self.model.save(filename)
        print(f"💾 모델이 '{filename}.zip'으로 저장되었습니다.")
        return True

    def load_model(self, filename="production_agent"):
        """저장된 모델 불러오기"""
        try:
            self.model = PPO.load(filename, env=self.env)
            self.is_trained = True
            print(f"📂 모델 '{filename}.zip'을 불러왔습니다.")
            return True
        except FileNotFoundError:
            print(f"❌ 모델 파일 '{filename}.zip'을 찾을 수 없습니다.")
            return False
        except Exception as e:
            print(f"❌ 모델 로딩 중 오류 발생: {e}")
            return False


# 메인 실행 코드
if __name__ == "__main__":
    print("🚀 생산라인 AI 에이전트 실행")
    print(f"🔩 설정된 최대 기계 수: {MAX_MACHINES_PER_STATION}대/스테이션")

    agent = SimpleProductionAgent()

    # 학습 실행 (시간을 늘려 더 나은 성능 기대)
    print("\n1️⃣ AI 학습 시작...")
    success = agent.train(total_timesteps=100000)

    if success:
        # 성능 테스트
        print("\n2️⃣ 학습된 AI 테스트...")
        agent.test_agent(num_tests=5)

        # 랜덤과 비교
        print("\n3️⃣ 성능 비교...")
        agent.compare_with_random(num_trials=10)

        # 모델 저장
        print("\n4️⃣ 모델 저장...")
        agent.save_model("my_production_ai")