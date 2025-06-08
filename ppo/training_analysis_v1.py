import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from rl_environment import ProductionLineEnv
from config import MAX_MACHINES_PER_STATION

# Set Korean font for plotting



class TrainingAnalyzer:
    """
    AI 학습 과정 분석 및 시각화 도구.
    config.py의 설정에 따라 동적으로 분석 범위를 조절하고,
    AI와 랜덤 에이전트의 성능을 심층적으로 비교 및 시각화합니다.
    """

    def __init__(self, model_path="my_production_ai"):
        self.env = ProductionLineEnv()
        self.model = self.load_trained_model(model_path)

    def load_trained_model(self, model_path):
        """Loads a trained AI model from the specified path."""
        try:
            model = PPO.load(model_path, env=self.env)
            print(f"✅ Successfully loaded model '{model_path}.zip'.")
            return model
        except FileNotFoundError:
            print(f"❌ Model '{model_path}.zip' not found. Please run simple_agent.py to train a model first.")
            return None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None

    def run_performance_analysis(self, num_episodes=100):
        """
        Gathers performance data for both AI and Random agents over multiple episodes.
        This data is the foundation for all comparative visualizations.
        """
        if not self.model:
            return None

        print(f"\n📊 Running performance analysis for AI vs. Random agent ({num_episodes} episodes each)...")

        agent_data = []
        for agent_type in ['AI', 'Random']:
            for i in range(num_episodes):
                obs, _ = self.env.reset()
                if agent_type == 'AI':
                    action, _ = self.model.predict(obs, deterministic=True)
                else:  # Random Agent
                    action = self.env.action_space.sample()

                _, reward, _, _, info = self.env.step(action)
                machines = action + 1

                agent_data.append({
                    'Agent': agent_type,
                    'Episode': i,
                    'Reward': reward,
                    'Throughput': info['throughput'],
                    'Cost': info['total_cost'],
                    'WaitTime': self.env.current_state[1],  # Get avg_wait_time from state
                    'Machining': machines[0],
                    'Assembly': machines[1],
                    'Inspection': machines[2],
                })

        print("✅ Analysis data collection complete.")
        return pd.DataFrame(agent_data)

    def print_summary_statistics(self, perf_df):
        """
        Prints a summary table with key statistics (mean, median, std) for AI and Random agents.
        Also calculates and displays percentage improvements of AI over Random.
        """
        if perf_df is None:
            return

        print("\n📈 Summary Statistics for AI vs. Random Agent")
        print("-" * 50)

        # Calculate statistics
        metrics = ['Reward', 'Throughput', 'Cost', 'WaitTime']
        stats = perf_df.groupby('Agent')[metrics].agg(['mean', 'median', 'std']).round(2)
        
        # Format the table
        print("\nSummary Table:")
        print(stats.to_string())

        # Calculate percentage improvements
        ai_means = stats.loc['AI', [(m, 'mean') for m in metrics]]
        random_means = stats.loc['Random', [(m, 'mean') for m in metrics]]
        
        print("\nPercentage Improvement of AI over Random:")
        print("-" * 50)
        for metric in metrics:
            ai_val = ai_means[(metric, 'mean')]
            random_val = random_means[(metric, 'mean')]
            if metric in ['Cost', 'WaitTime']:  # Lower is better
                improvement = ((random_val - ai_val) / random_val) * 100 if random_val != 0 else 0
            else:  # Higher is better
                improvement = ((ai_val - random_val) / random_val) * 100 if random_val != 0 else 0
            print(f"{metric}: {improvement:.2f}%")

    # 기존 코드의 plot_performance_dashboard 함수를 아래 코드로 교체하세요.

    def plot_performance_dashboard(self, perf_df):
        """
        (최종 수정 버전 v3)
        - 가장 정교한 비교 기준(최대 생산량 그룹 내 최저 비용)을 적용합니다.
        - 핵심 성과 지표는 터미널에 텍스트로 출력합니다.
        - 4개의 핵심 그래프를 2x2 그리드에 최적화하여 배치하고 가독성을 높입니다.
        """
        if perf_df is None:
            return
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        print("🎨 최종 버전의 대시보드 생성을 시작합니다 (결과는 터미널과 차트 창에 표시됩니다)...")

        # === 데이터 준비 및 핵심 지표 계산 (가장 정교한 비교 기준 적용) ===
        ai_data = perf_df[perf_df['Agent'] == 'AI']
        random_data = perf_df[perf_df['Agent'] == 'Random']

        # AI 평균 KPI
        ai_kpi = ai_data[['Throughput', 'Cost', 'WaitTime']].mean()
        # 랜덤 평균 KPI (차트용)
        random_avg_kpi = random_data[['Throughput', 'Cost', 'WaitTime']].mean()
        
        # === [수정] '최고의 랜덤 방식' 정의 및 해당 비용 계산 ===
        # 1. 랜덤 방식의 최대 생산량(Throughput)을 찾습니다.
        max_throughput_for_random = random_data['Throughput'].max()
        # 2. 최대 생산량을 기록했던 모든 경우를 필터링합니다.
        best_random_sessions = random_data[random_data['Throughput'] == max_throughput_for_random]
        # 3. 그중에서 가장 비용(Cost)이 낮았던 값을 찾습니다.
        best_random_cost = best_random_sessions['Cost'].min()

        # 성과 개선율 계산
        throughput_improvement = ((ai_kpi['Throughput'] - random_avg_kpi['Throughput']) / random_avg_kpi['Throughput']) * 100
        cost_reduction_vs_best = ((best_random_cost - ai_kpi['Cost']) / best_random_cost) * 100
        wait_time_reduction = ((random_avg_kpi['WaitTime'] - ai_kpi['WaitTime']) / random_avg_kpi['WaitTime']) * 100

        # === [수정] 핵심 성과 지표를 터미널에 출력 ===
        print("\n" + "="*60)
        print("      AI 기반 생산 라인 최적화 성과 보고서 (Terminal Output)")
        print("="*60)
        print(f"\n[핵심 성과 요약]\n")
        print(f"• 시간당 생산량: AI 제어 방식이 랜덤 방식 평균 대비 {throughput_improvement:.1f}% 더 많이 생산했습니다.")
        print(f"• 운영 비용 효율성: AI의 평균 비용은, 랜덤 방식이 '최대 생산량'을 냈을 때의 '최저 비용'보다 {cost_reduction_vs_best:.1f}% 더 낮았습니다.")
        print(f"• 평균 대기 시간: AI 제어 방식이 랜덤 방식 평균 대비 {wait_time_reduction:.1f}% 더 단축시켰습니다.")
        print("\n[결론]")
        print("AI는 생산량을 극대화하면서 동시에 비용을 최적화하는, 더 지능적인 운영 전략을 수행합니다.")
        print("="*60 + "\n")

        # === 시각화 설정 ===
        palette = {'AI': '#007bff', 'Random': '#adb5bd'}
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        
        # === [수정] Figure 및 2x2 그리드 레이아웃 설정 ===
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('AI 기반 생산 라인 최적화 성과 분석', fontsize=24, fontweight='bold')

        kpi_data = perf_df.groupby('Agent')[['Throughput', 'Cost', 'WaitTime']].mean().reset_index()

        # --- 2x2 차트 영역 ---
        # 1. 시간당 생산량 비교
        ax1 = axes[0, 0]
        sns.barplot(data=kpi_data, x='Agent', y='Throughput', hue='Agent', ax=ax1, palette=palette, width=0.6, legend=False)
        ax1.set_title('시간당 생산량(Throughput) 비교', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('개/시간', fontsize=12)
        ax1.set_xlabel('')
        # Tick 위치를 명확하게 설정
        ax1.set_xticks(range(len(kpi_data)))
        ax1.set_xticklabels(['AI 제어', '랜덤 방식'])
        for p in ax1.patches:
            ax1.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=11, color='black', xytext=(0, 8),
                        textcoords='offset points')

        # 2. 평균 운영 비용 비교
        ax2 = axes[0, 1]
        sns.barplot(data=kpi_data, x='Agent', y='Cost', hue='Agent', ax=ax2, palette=palette, width=0.6, legend=False)
        ax2.set_title('평균 운영 비용(Cost) 비교', fontsize=16, fontweight='bold', pad=20)
        ax2.set_ylabel('$', fontsize=12)
        ax2.set_xlabel('')
        # Tick 위치를 명확하게 설정
        ax2.set_xticks(range(len(kpi_data)))
        ax2.set_xticklabels(['AI 제어', '랜덤 방식'])
        for p in ax2.patches:
            ax2.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=11, color='black', xytext=(0, 8),
                        textcoords='offset points')

        # 3. 평균 대기 시간 비교
        ax3 = axes[1, 0]
        sns.barplot(data=kpi_data, x='Agent', y='WaitTime', hue='Agent', ax=ax3, palette=palette, width=0.6, legend=False)
        ax3.set_title('평균 대기 시간(Wait Time) 비교', fontsize=16, fontweight='bold', pad=20)
        ax3.set_ylabel('분', fontsize=12)
        ax3.set_xlabel('')
        # Tick 위치를 명확하게 설정
        ax3.set_xticks(range(len(kpi_data)))
        ax3.set_xticklabels(['AI 제어', '랜덤 방식'])
        for p in ax3.patches:
            ax3.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=11, color='black', xytext=(0, 8),
                        textcoords='offset points')
        
        # 4. 운영 전략 분포
        ax4 = axes[1, 1]
        sns.scatterplot(data=perf_df, x='Cost', y='Throughput', hue='Agent', ax=ax4, palette=palette, alpha=0.7, s=60, edgecolor='w', linewidth=0.5)
        ax4.set_title('운영 전략 분포 (비용 vs 생산량)', fontsize=16, fontweight='bold', pad=20)
        ax4.set_xlabel('총 운영 비용 ($)', fontsize=12)
        ax4.set_ylabel('시간당 생산량 (개)', fontsize=12)
        ax4.legend(title='제어 방식', loc='lower right')
        ax4.text(0.95, 0.95, 'AI 목표 영역\n(저비용, 고생산성)', transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', ec='lightgray'))

        # === [수정] tight_layout 대신 subplots_adjust로 그래프 간 간격 세밀하게 조정 ===
        plt.subplots_adjust(left=0.07, right=0.95, top=0.88, bottom=0.08, hspace=0.35, wspace=0.2)
        plt.show()

# Main execution code
if __name__ == "__main__":
    print("📊 AI 학습 결과 분석 도구")
    print(f"🔩 설정된 최대 기계 수: {MAX_MACHINES_PER_STATION}대/스테이션")
    print("-" * 50)

    # 1. Create analyzer and load model
    analyzer = TrainingAnalyzer(model_path="my_production_ai")

    if analyzer.model:
        # 2. Collect performance data
        performance_data = analyzer.run_performance_analysis(num_episodes=200)

        # 3. Print summary statistics
        if performance_data is not None:
            analyzer.print_summary_statistics(performance_data)

        # 4. Visualize performance dashboard
        if performance_data is not None:
            analyzer.plot_performance_dashboard(performance_data)

        print("\n🎉 모든 분석이 완료되었습니다.")