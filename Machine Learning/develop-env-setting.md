# 개발환경 세팅

Windows 기준으로 설명

## Anaconda
수치해석 관련 등 라이브러리가 포함된 Anaconda를 이용. 기존 파이썬 버전 3.6에서는 Tensorflow가 호환되지 않으므로 최신버전 Anaconda로는 tf를 사용할 수 없음

[Anaconda3 4.2.0](https://repo.continuum.io/archive/Anaconda2-4.2.0-Windows-x86_64.exe)

## CUDA
tensorflow gpu의 설치를 위해서는 CUDA 라이브러리를 설치해주어야한다

[CUDA 8.0](https://developer.nvidia.com/compute/cuda/8.0/Prod2/network_installers/cuda_8.0.61_win10_network-exe)과 [CUDNN 5.0](https://developer.nvidia.com/rdp/cudnn-download#collapseTwo)를 다운로드한 후 CUDA 설치하고, CUDA 설치 폴더(기본: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0)에 CUDNN의 압축을 푼 후 해당 파일들을 붙여넣는다

** CUDNN도 최신버전을 쓰면 아래와 같은 에러가 나며서 실행되지 않는다
![DLL load failed](http://i.imgur.com/3tYkPkm.png)

## tensorflow gpu
아래와 같은 명령어를 이용하여 설치를 진행한다

`
    pip install -U tensorflow-gpu
`