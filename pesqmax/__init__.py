"""PESQ-WB 극대화 음성 향상 파이프라인.

구성: MP-SENet(Non-causal TS-Conformer) + 학습 가능한 PCS 전/후처리 +
MetricGAN+ 스타일 PESQ 판별자 + 미분 가능한 PESQ 손실(torch-pesq 이식본).
"""

__version__ = "1.0.0"
