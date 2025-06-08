# simple_agent_with_plot.py

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import numpy as np
# stable_baselines3에서 제공하는 Monitor를 임포트하여 보상을 쉽게 기록합니다.
from stable_baselines3.common.monitor import Monitor
# 그래프 생성을 위해 matplotlib.pyplot을 임포트합니다.
import matplotlib.pyplot as plt

# rl_environment.py와 config.py 파일은 동일한 디렉토리에 있어야 합니다.
from rl_environment import ProductionLineEnv
from config import MAX_MACHINES_PER_STATION

class SimpleProductionAgent:
    """
    동적 생산라인 환경에 맞춰 학습하는 AI 에이전트.
    에피소드별 보상을 기록하고 학습 완료 후 그래프로 시각화하는 기능이 추가되었습니다.
    """

    def __init__(self):
        # Monitor로 환경을 감싸주면 에피소드별 보상, 스텝 수 등이 자동으로 기록됩니다.
        self.env = Monitor(ProductionLineEnv())
        self.model = None
        self.is_trained = False

    def train(self, total_timesteps):
        """AI 에이전트 학습"""
        print("🔍 환경 검증 중...")

        try:
            check_env(self.env)
            print("✅ 환경 검증 완료!")
        except Exception as e:
            print(f"❌ 환경 오류: {e}")
            return False

        print(f"🧠 AI 학습 시작 (총 {total_timesteps} 스텝)")

        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=1024,
            batch_size=64,
            gamma=0.99,
            ent_coef=0.01,
            clip_range=0.2,
            tensorboard_log="./ppo_production_tensorboard/"
        )

        self.model.learn(total_timesteps=total_timesteps)
        self.is_trained = True

        print("🎉 학습 완료!")
        return True

    def plot_training_rewards(self):
        """학습 과정에서 기록된 에피소드별 보상을 그래프로 생성합니다."""
        if not self.is_trained:
            print("❌ 먼저 학습을 완료해주세요!")
            return

        # Monitor 환경에서 get_episode_rewards()를 통해 모든 에피소드의 보상 기록을 가져옵니다.
        rewards = self.env.get_episode_rewards()

        if not rewards:
            print("⚠️ 보상 기록을 찾을 수 없습니다. 학습 시간이 너무 짧았을 수 있습니다.")
            return

        print(f"📊 총 {len(rewards)}개 에피소드에 대한 보상 그래프를 생성합니다.")

        # 이동 평균을 계산하여 학습 추세를 더 명확하게 확인합니다.
        window = 100
        if len(rewards) < window:
            window = len(rewards) // 5 if len(rewards) > 10 else 1

        moving_average = []
        if window > 0:
            moving_average = np.convolve(rewards, np.ones(window)/window, mode='valid')

        plt.figure(figsize=(12, 6))
        plt.plot(rewards, label='Episodic Reward', alpha=0.5)
        if len(moving_average) > 0:
            # 이동 평균 그래프의 시작점을 맞추기 위해 x축 인덱스를 조정합니다.
            plt.plot(np.arange(window-1, len(rewards)), moving_average,
                     label=f'Moving Average (window={window})', color='red', linewidth=2)

        # 요청대로 그래프의 제목과 레이블을 영문으로 설정합니다.
        plt.title('Training Progress: Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend(loc='upper left')
        plt.grid(True)

        # 생성된 플롯을 이미지 파일로 저장합니다.
        plt.savefig("training_reward_plot.png")
        print("✅ 보상 그래프가 'training_reward_plot.png' 파일로 저장되었습니다.")

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

        # 테스트 시에는 학습 기록에 영향을 주지 않도록 새로운 환경 인스턴스를 사용합니다.
        test_env = ProductionLineEnv()

        for test_num in range(num_tests):
            obs, _ = test_env.reset()
            action, _ = self.model.predict(obs, deterministic=True)
            machines = action + 1

            next_obs, reward, terminated, truncated, info = test_env.step(action)

            results.append({
                'test_num': test_num + 1,
                'action': machines,
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

        test_env = ProductionLineEnv()
        ai_rewards, ai_throughputs = [], []
        for _ in range(num_trials):
            obs, _ = test_env.reset()
            ai_action, _ = self.model.predict(obs, deterministic=True)
            _, reward, _, _, info = test_env.step(ai_action)
            ai_rewards.append(reward)
            ai_throughputs.append(info['throughput'])

        avg_ai_reward = np.mean(ai_rewards)
        avg_ai_throughput = np.mean(ai_throughputs)

        random_rewards, random_throughputs = [], []
        for _ in range(num_trials):
            obs, _ = test_env.reset()
            random_action = test_env.action_space.sample()
            _, reward, _, _, info = test_env.step(random_action)
            random_rewards.append(reward)
            random_throughputs.append(info['throughput'])

        avg_random_reward = np.mean(random_rewards)
        avg_random_throughput = np.mean(random_throughputs)

        obs, _ = test_env.reset()
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

    # 빠른 테스트를 위해 학습 스텝 수를 20,000으로 설정했습니다.
    # 더 높은 성능을 원하시면 100,000 이상으로 값을 높여서 실행해보세요.
    print("\n1️⃣ AI 학습 시작...")
    success = agent.train(total_timesteps=200000)

    if success:
        # 2️⃣ 학습 완료 후 보상 그래프 생성
        print("\n2️⃣ 학습 보상 그래프 생성...")
        agent.plot_training_rewards()

        # 3️⃣ 학습된 AI 테스트
        print("\n3️⃣ 학습된 AI 테스트...")
        agent.test_agent(num_tests=5)

        # 4️⃣ 랜덤 에이전트와 성능 비교
        print("\n4️⃣ 성능 비교...")
        agent.compare_with_random(num_trials=10)

        # 5️⃣ 학습된 모델 저장
        print("\n5️⃣ 모델 저장...")
        agent.save_model("my_production_ai")