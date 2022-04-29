## Deep Image Alignment

### 使用

#### Training

```shell
./tools/dist_train.sh configs/udis_iter60w.py 8
```

#### Evaluate

```shell
# 生成Homography Matrix
python tools/test.py \
       configs/udis_iter60w.py \  # config文件
       work_dirs/pt-best/iter_160000.pth \  # 保存的模型
       --out work_dirs/pt_iter160000.pkl  # 保存H_matrix文件的地址
# 使用重投影误差衡量Homography Matrix
python tools/evaluate.py \
       work_dirs/pt_iter160000.pkl \  # 保存H的文件
       data/udis-d/testing_infos.pkl \  # 数据集信息
       --result-save-dir work_dirs/pt_iter160000 \  # 保存变换结果路径，不保存可以不设
       --blending-scale 1.0     # overlap区域混合系数，默认0.5，1.0表示只用img1 
```

传统算法baseline

```shell
python tools/baseline_sift.py \
       data/udis-d/testing_infos.pkl \  # 数据集信息
       work_dirs/udis-sift-baseline.pkl # 保存H_matrix文件的地址
```

#### 其它功能

切换tensorflow版的spatial transform

```shell
export USE_TF_STL=1
```

切换成功会打印“Spatial transform layer using tensorflow-version!!'”

默认使用pytorch版spatial transform，训练速度会比tf版快15%

### 更新日志

###### 2022-04-25

* 添加CCL(Contextual Correlation Layer)和fast CCL(比CCL快50%)

###### 2022-04-21

* 支持使用不同的feature_extractor

###### 2022-04-20

* 修复pytorch spatial transform导致loss收敛异常的问题

* 添加切换tensorflow版spatial transform的环境变量

###### 2022-04-14

* 完善模型测试流程

* 添加单应性矩阵评价脚本

* 添加SIFT-RANSAC Baseline
