Path Planning Algorithms
========================

1\. 개요 및 구조
-----------

이 프로젝트는 기존의 경로 계획 알고리즘을 통합에 맞도록 업데이트한 버전.  
`Image2Plan.py` 파일에서 경로계획 기본 정보를 전송하여, `Plan2WP.py`에서 경로 계획을 진행.  
'custom_msgs'를 통해 이미지 경로, 시작점과 도착점, Output Mode를 전송.  


*   **Image2Plan.py**: 이미지 경로, 시작점과 도착점, Output Mode를 메시지로 전송.
*   **Plan2WP.py**: 경로 계획 결과를 PNG 그림 파일로 생성, 이전 알고리즘과 비교하여 계산된 Cost를 출력.

2\. 필요 작업
---------

프로젝트를 정상적으로 실행하기 위해서는 'pathPlanning\_data' 폴더의 'ros2_ws' 폴더를 기존 'ros2_ws'로 옮겨야함.  
이 폴더는 다음과 같은 내용을 포함해야 합니다 :

*   `src` 폴더
*   `90_exp_263k.onnx`
*   `1000-001.png`
*   `test26.onnx`
*   **pathPlanning_data** 폴더는 NAS1 '무인이동체_외부공유/유창경교수님/pathPlanning_data'에 있음.
*   "https://acslc.synology.me/sharing/q6XLjNuyc"


3\. 실행
-----------


프로젝트를 실행하는 커맨드는 다음과 같음.

**터미널 - 1: 빌드**

`cd ~/ros2_ws`   
`colcon build`

**터미널 - 2: 경로계획 기본 정보 Publish**

`source install/setup.bash`  
`ros2 run py_pubsub talker`

**터미널 - 3: 경로계획 기본 정보 Subscribe & 경로계획 Main문**

`source install/setup.bash`  
`ros2 run py_pubsub listener`


4\. 업데이트 내역
-----------



24.01.01 : RRT 알고리즘과 학습 onnx모델을 비교하는 mode = 3 추가. (기본값으로 설정)
