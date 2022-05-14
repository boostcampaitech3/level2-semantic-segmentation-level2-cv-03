# 🌏 Semantic Segmentation for Recycling Trash
<br/>  

## 🎇 Main Subject
대량 생산, 대량 소비의 시대에서는 필연적으로 “쓰레기 처리”문제가 발생합니다. 분리 수거는 이러한 환경 부담을 줄일 수 있는 대표적인 방법이며, 올바른 방식으로 분리수거를 수행해야 합니다.

해당 프로젝트에서는 사진에서 쓰레기를 segmentation하는 모델을 만들어 분리수거를 진행하고자 하였고 특히 11가지로 나뉘는 쓰레기 종류와 위치를 파악하기 위한 모델을 만드는 것에 집중하였습니다.

<br/>  
  
## 💻 Development Environment
**개발 언어** : PYTHON (IDE: VSCODE, JUPYTER NOTEBOOK)

**서버**: AI STAGES (GPU: NVIDIA TESLA V100)

**협업 Tool** : git, notion, [wandb](https://wandb.ai/cv-3-bitcoin), [google spreadsheet](https://docs.google.com/spreadsheets/d/174jHw0l98ar1yy-vPYu4Vh6XP_bMv1H1/edit#gid=1619052354), slack

**Library** : mmsegmentation, smp 
   
<br/>  
  
## 🌿 Project Summary
  - **Data Augmentation**
    - Horizontal Flip
    - Rotate90
    - GridDropOut
    - RandomResizedCrop
  - **TTA**
    - Inference(Test) 과정에서 Augmentation 을 적용한 뒤 예측의 확률을 평균(또는 다른 방법)을 통해 도출하는 기법
    - Multiscale → 0.5, 0.75, 1.0, 1.25, 1.5의 ratio를 사용
    - Flip → Horizontal & Vertical
  - **Ensemble**

### Dataset
  - 재활용 쓰레기가 촬영된 **.jpg 형식의 이미지**와 **masking되어 있는 고유 좌표** 및 종류를 명시한 **.json 파일**로 이루어져 있으며 각각 train, val, test로 구분
  - **범주** : 배경, 일반쓰레기, 종이, 종이팩, 금속, 유리, 플라스틱, 스티로폼, 플라스틱 가방, 배터리, 의류 (총 11가지)
### Metrics
  - **mIoU(Mean Intersection over Union)**
    - Semantic Segmentation에서 사용하는 대표적인 성능 측정 metric
    - GT값과 prediction값의 class별 IoU의 평균을 계산
### Model
|Model|Backbone|library|LB Score@public|LB Score@private|
|:---:|:---:|:---:|---:|---:|
|KNet + UperNet|Swin-L|mmsegmentation|0.7083|0.7245|
|DeepLabV3Plus|xception65|Segmentation Models Pytorch(SMP)|0.6249|0.6102|
|DeepLabV3Plus|EfficientNet-b7|SMP|0.6173|0.5755|
|Unet|EfficientNet-b7|SMP|0.6463|0.6429|
|Unet|regnet|SMP|0.6548|0.6265|
|Deeplab v3|ResNet50|base|0.5454|0.5225|
|UperNet|ResNet101|base|0.6319|0.5839|
<br/>  

## [Wrap Up Report](https://sand-bobolink-9c4.notion.site/Wrap-Up-4a59a89080a34b9b91c1ec0cc5ad8d40)
