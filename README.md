# jeju-stonewall-lbm

제주 돌담처럼 돌 사이에 구멍이 뚫린 담이 바람에 왜 덜 넘어지는지, 틈 없는 담과 비교한 격자 볼츠만(LBM) 유체 시뮬레이션입니다.

- **3D:** 불규칙하게 쌓은 돌담(공극률 0%, 20%, 40%)을 D3Q19 + LES로 계산합니다. NVIDIA Warp로 작성해 GPU에서 돕니다.
- **2D:** 처음 만든 단면 버전입니다. D2Q9 + LES로 계산하고, NumPy만 씁니다.

## 핵심 결과 (3D, H = 40격자)

| | 평균 항력 C<sub>d</sub> | 돌풍 최대 C<sub>d</sub> | 힘의 출렁임 | 최대 전도 모멘트 | 담 뒤 역류 구역 |
|---|---|---|---|---|---|
| 틈 없는 담 | 1.40 | 1.97 | 0.174 | 1.05 | 6.1 H² |
| 돌담 20% | 1.23 (−12%) | 1.39 (−30%) | 0.056 (−68%) | 0.71 (−33%) | 6.4 H² |
| 돌담 40% | 0.83 (−41%) | 1.02 (−49%) | 0.067 (−61%) | 0.51 (−52%) | 0.6 H² |

구멍은 평균 힘보다 돌풍 때의 최대 힘과 그 출렁임을 훨씬 크게 줄입니다. 돌이 적어 가벼워지는 것까지 따지면, 무게 대비 넘어뜨리는 힘은 12~17% 줄어듭니다. 자세한 분석과 한계는 [RESULTS.md](RESULTS.md)에 있습니다.

## 결과 뷰어

`results/`의 HTML 파일은 데이터가 모두 들어 있는 단일 파일입니다. 내려받아 브라우저로 열면 됩니다. GitHub 화면에서는 HTML이 그려지지 않습니다.

| 파일 | 내용 |
|---|---|
| `results/3d/stone_wall_scene3d.html` | 마우스로 돌려 보는 3D 장면: 돌, 평균 유선과 흐르는 입자, 역류 구역, 시간에 따라 재생되는 3D 소용돌이(Q-criterion 볼륨). three.js를 CDN에서 불러오므로 인터넷 필요 |
| `results/3d/stone_wall_3d.html` | 옆 단면, 높이 H/2 수평 단면 애니메이션과 시간에 따른 항력 그래프, 비교표 |
| `results/2d/stone_wall_wind.html` | 2D 계산의 소용돌이 애니메이션과 항력 그래프 |
| `results/2d/*.png`, `*.gif` | 2D 계산의 평균 유선, 압력 분포, 힘 비교, 소용돌이 GIF |

## 폴더 구성

```
stonewall3d/   3D 솔버, 돌담 생성기, 후처리, 뷰어 템플릿 (실행법은 stonewall3d/README.md)
2d/            2D 솔버와 후처리
results/       계산 결과 뷰어
RESULTS.md     결과 분석
```

## 다시 계산하기

**3D** (NVIDIA GPU, VRAM 약 8 GB, 세 경우 합쳐 약 35분 @ RTX 5070 Ti):

```bash
cd stonewall3d
pip install warp-lang numpy matplotlib pillow scikit-image
bash run_all.sh                 # results_H40/ 에 결과와 뷰어 생성
```

5000스텝마다 체크포인트를 저장하므로, 중간에 끊겨도 같은 명령으로 이어서 돌릴 수 있습니다.

**2D** (CPU, 경우당 수 분):

```bash
cd 2d
python lbm_wall.py solid 0 0    # 틈 없는 담
python lbm_wall.py p20 4 2      # 가로 틈 4개, 공극률 20%
python lbm_wall.py p40 4 4      # 가로 틈 4개, 공극률 40%
python analyze.py solid p20 p40 # 그림과 GIF(ffmpeg 필요)
python export_html.py solid p20 p40   # stone_wall_wind.html
```

결과 원본(`.npz`)은 경우당 약 250 MB로, GitHub 파일 크기 제한(100 MB)을 넘어서 저장소에 넣지 않았습니다.
