# 3D 돌담 바람 시뮬레이션

불규칙하게 쌓은 돌담(공극률 20%, 40%)과, 같은 돌에 틈을 메운 담(0%)에 같은 바람을 불어 넣고 담이 받는 힘을 비교하는 3D 격자 볼츠만(LBM) 코드입니다. NVIDIA Warp로 작성해서 같은 코드가 GPU(`--device cuda`)와 CPU(`--device cpu`, 느림)에서 모두 돕니다.

## 파일

| 파일 | 내용 |
|---|---|
| `lbm3d.py` | 솔버. D3Q19 정규화 BGK + Smagorinsky LES, 운동량 교환법으로 매 스텝 담에 걸리는 힘 기록 |
| `geometry.py` | 돌담 생성기. 돌을 한 줄씩 아래 돌 위에 떨어뜨려 촘촘히 쌓은 뒤, 각 돌을 제자리에서 줄여 공극률을 맞춤. 꼭대기 선을 이루는 돌은 윗면을, 맨 아래 돌은 지면 접촉을 고정해서 모든 공극률에서 담의 최고 높이가 같음 |
| `postprocess.py` | 결과(`.npz`)로 인터랙티브 HTML 뷰어, GIF, 힘 비교 그림 생성 |
| `viewer_template.html` | 뷰어 틀 (`postprocess.py`가 사용) |
| `run_all.sh` | 세 경우 실행 + 후처리 한 번에 |

## 설치 (WSL2)

```bash
conda create -n stonewall python=3.12 -y && conda activate stonewall
pip install warp-lang numpy matplotlib pillow
python -c "import warp as wp; wp.init()"      # 장치 목록에 "cuda:0" : NVIDIA GeForce RTX 5070 Ti 가 보여야 함
```

RTX 50 시리즈(Blackwell)는 최신 NVIDIA 드라이버와 최신 `warp-lang`이 필요합니다. `wp.init()`에 GPU가 안 보이면 Windows 쪽 드라이버를 먼저 업데이트하세요.

## 실행

먼저 1~2분짜리 확인용 실행:

```bash
python lbm3d.py --case quick --porosity 0.2 --H 16 --t-spinup 2 --t-avg 2 --out quick
```

로그의 `MLUPS`(초당 백만 격자 갱신)로 속도를 확인한 뒤, 본 계산:

```bash
bash run_all.sh                 # H=40, 격자 800×160×321 (4100만 격자), VRAM 약 8 GB
H=32 bash run_all.sh            # 더 가볍게, VRAM 약 4 GB
```

결과는 `results_H40/`에 생깁니다. `stone_wall_3d.html`을 브라우저로 열면 됩니다.

### 중간에 끊겼을 때

5000스텝(약 1.5분)마다 `results_H40/<case>.ckpt/`에 체크포인트가 저장됩니다. 끊기면 **같은 명령을 다시 실행**하면 됩니다. `run_all.sh`는 끝난 경우를 건너뛰고, 끊긴 경우는 마지막 체크포인트부터 이어서 돌립니다. 이어서 돌린 결과는 한 번에 돌린 결과와 비트 단위로 같습니다. 체크포인트 하나는 H=40에서 약 4 GB이고, 그 경우가 끝나면 자동으로 지워집니다.

- 옵션을 바꿔서 다시 돌리면 체크포인트와 설정이 달라 멈춥니다. 처음부터 하려면 `EXTRA=--no-resume bash run_all.sh`
- 저장 간격 바꾸기: `EXTRA="--ckpt-every 2000" bash run_all.sh`, 끄기: `--ckpt-every 0`
- 터미널이나 SSH가 끊겨도 계속 돌게 하려면 `tmux` 안에서 실행하세요: `tmux new -s wall` → `bash run_all.sh` → `Ctrl-b d`로 빠져나오기, 다시 볼 때 `tmux attach -t wall`
- 결과 폴더는 Windows 쪽(`/mnt/c/...`)이 아니라 WSL 홈(`~/...`)에 두세요. 체크포인트 저장이 훨씬 빠릅니다.

### 소요 시간 가늠

한 경우의 스텝 수는 (`--t-spinup` + `--t-avg`) × H / U_H 이고, 기본값(20 + 30 H/U, H=40)이면 약 29,000 스텝입니다. 시간 = 스텝 수 × 격자 수 / (MLUPS × 10⁶) 이므로, 확인용 실행에서 나온 MLUPS를 넣어 계산해 보세요. 예를 들어 1000 MLUPS라면 경우당 약 20분입니다.

## 주요 옵션

| 옵션 | 기본값 | 뜻 |
|---|---|---|
| `--H` | 40 | 담 높이의 격자 수 (해상도). 메모리는 H³에 비례 |
| `--porosity` | (필수) | 바람 방향으로 들여다봤을 때 뚫려 보이는 면적 비율. 0이면 틈을 메운 담 |
| `--seed` | 7 | 돌 배치. 비교할 경우끼리는 같게 둘 것 |
| `--Re` | 15000 | U_H·H/ν |
| `--up` `--down` `--span` `--height` | 4, 16, 4, 8 | 계산 영역 크기 (H 단위). 담 길이 방향(`span`)은 주기 경계 |
| `--thick` | 0.3 | 담 두께 (H 단위) |
| `--t-spinup` `--t-avg` | 20, 30 | 흐름이 자리 잡는 시간, 평균·기록 시간 (H/U_H 단위) |
| `--snap-dt` | 0.25 | 애니메이션 프레임 간격 (H/U_H 단위) |

## 모델 요약

- 축: x 바람 방향, y 담 길이 방향(주기), z 높이. 맨 아래 격자층이 지면.
- 유입: 높이 3H까지 u ∝ z^(1/7)인 지표 바람. 기준 풍속 U_H는 담 꼭대기 높이의 유입 풍속.
- 경계: 지면·돌은 half-way bounce-back, 위는 미끄럼(free-slip), 출구 앞 3H는 점성을 키운 흡수층.
- 힘: 운동량 교환법. 매 스텝 높이별로 기록해서 밑동 기준 전도 모멘트도 계산.
- 계수: C_d = F_x / (½ρU_H²·H·L), C_m = M / (½ρU_H²·H²·L), L = 계산한 담 길이.

## 한계

- 유입 바람에 난류 성분이 없습니다. 실제 돌풍은 바람 자체에도 섞여 들어옵니다.
- Re는 실제 바람(~10⁶)보다 작고, 벽 근처를 모델링하는 벽 함수 없이 격자로만 풉니다.
- 계산 영역 높이 8H라 막힘 비율은 1/8입니다. 이전 2D 계산(1/4)보다는 작지만 0은 아닙니다.

## 3D로 돌려 보기 (`export3d.py`)

```bash
pip install scikit-image
python export3d.py results_H40/solid.npz results_H40/p20.npz results_H40/p40.npz --out results_H40
```

`results_H40/stone_wall_scene3d.html`이 생깁니다. 브라우저에서 열고 마우스로 돌려 볼 수 있습니다(three.js를 CDN에서 불러오므로 인터넷 필요).

- **평균 흐름:** 돌 메쉬, 담 앞에서 출발해 틈을 지나는 평균 유선, 유선을 따라 흐르는 입자, 시간 평균으로 바람이 거꾸로 부는 역류 구역. 모든 결과에서 나옵니다.
- **3D 소용돌이 재생:** 순간 바람장의 Q-criterion을 볼륨으로 그려 시간에 따라 재생합니다. 3D 프레임을 저장하는 현재 `lbm3d.py`로 계산한 결과에만 있습니다(`--snap3d-every`, 기본 0.5 H/U 간격).
- `run_all.sh`는 마지막에 이 스크립트도 자동으로 실행합니다.
