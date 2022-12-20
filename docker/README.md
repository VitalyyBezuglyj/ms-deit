Создание образа из докер файла:
```bash
./build.sh
```

Запуск конейнера:
```bash
./start.sh
```

Подключение к контейнеру:
```bash
./into.sh
```


## Конфигурирование

Файлы конфигурации находятся в директории [config](../config).

1. Для запуска необходимо настроить параметры камеры, начинающиеся с `Camera.`
2. От параметра `Feature.max_num_keypoints` сильно зависит качество/производительность метода. Увеличение всегда приводит к падению производительности. При этом, как правило, улучшается качество, но может и не измениться или даже упасть.
3. Параметры tracking_method_X определяют порядок вызова методов трекинга. При неуспешном завершении одного вызывается следующий. motion - самый быстрый, но может вызвать срыв трекинга даже при успешном завершении. robust - самый медленный и надежный.
4. Параметр `gpu` определяет номер устройства, используемого в алгоритме локализации.
5. Парметры `num_last_frames`, `take_every_frame` определяют стратегию выбора кадров для локализации. Увеличение `num_last_frames` приводит к замедлению метода, но, как правило, к увеличению качества. `take_every_frame` определяет промежуток между кадрами. Если `take_every_frame: 10` и частота кадров `Camera.fps: 10` то будет использоваться один кадр в секунду. Если при этом `num_last_frames: 5`, то будут использовать данные за последние 5 секунд.

## Сборка

Для целей отладки рекомендуется собирать с флагом `-DUSE_PANGOLIN_VIEWER=ON`, который добавляет запуск Pangolin Viewer, отображающий процесс работы метода.

Сначала необходимо собрать библиотеку libopenvslam, а затем ROS узлы, её использующие.

### Сборка libopenvslam и примеров

```bash
cd /home/docker_openvslam/catkin_ws/src/openvslam
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_WITH_MARCH_NATIVE=ON \
      -DUSE_PANGOLIN_VIEWER=OFF \
      -DUSE_SOCKET_PUBLISHER=OFF \
      -DUSE_STACK_TRACE_LOGGER=ON \
      -DBOW_FRAMEWORK=DBoW2 \
      -DBUILD_TESTS=OFF \
      ..
```

Для отладки
```bash
make -j4
```

Результаты сборки (библиотеки, исполняемые файлы) будут расположены в директории `/home/docker_openvslam/catkin_ws/src/openvslam/build`

Для релиза
```bash
sudo make -j4 install
sudo ldconfig
cd .. && rm -rf build
```

Библиотека и заголовочные файлы будут установлены в `/usr/local`.

### Сборка ROS узлов

Для отладки
```bash
source /opt/ros/melodic/setup.bash
cd /home/docker_openvslam/catkin_ws
catkin_make -DBUILD_WITH_MARCH_NATIVE=ON \
            -DUSE_PANGOLIN_VIEWER=OFF \
            -DUSE_SOCKET_PUBLISHER=OFF \
            -DUSE_STACK_TRACE_LOGGER=ON \
            -DBOW_FRAMEWORK=DBoW2 \
            -DBUILD_TESTS=OFF \
            -DCMAKE_BUILD_TYPE=Release \
            --only-pkg-with-deps openvslam \
            -j4
source devel/setup.bash
```

Для релиза
```bash
source /opt/ros/melodic/setup.bash
cd /home/docker_openvslam/catkin_ws
catkin_make install \
            -DBUILD_WITH_MARCH_NATIVE=ON \
            -DUSE_PANGOLIN_VIEWER=OFF \
            -DUSE_SOCKET_PUBLISHER=OFF \
            -DUSE_STACK_TRACE_LOGGER=ON \
            -DBOW_FRAMEWORK=DBoW2 \
            -DBUILD_TESTS=OFF \
            -DCMAKE_BUILD_TYPE=Release \
            --only-pkg-with-deps openvslam \
            -j4
rm -rf build devel
source install/setup.bash
```

### Запуск standalone

Исходный код примеров запуска расположен в [example](../example). 

Далее предполагается, что данные извлечены в формате TUM и расположены в `path/to/dataset/in/tum/format`.

Пример запуска SLAM в режиме RGBD на датасете SDBCS Husky
```bash
cd /home/docker_openvslam/catkin_ws/src/openvslam/build
./run_tum_rgbd_slam \
            -v /home/docker_openvslam/catkin_ws/src/openvslam/orb_vocab/orb_vocab.dbow2 \
            -d path/to/dataset/in/tum/format/06_mix_1_2020-03-17-14-47-54 \
            -c /home/docker_openvslam/catkin_ws/src/openvslam/config/husky_sdbcs_rgbd/Husky_SDBCS_03_17.yaml \
            --eval-log \
            --output_poses 06_mix_1_2020-03-17-14-47-54.txt
```
При этом полученная траектория будет сохранена в TUM формате в файл `06_mix_1_2020-03-17-14-47-54.txt` в директории запуска.

Пример запуска локализации а режиме RGBD на датасете SDBCS Husky
```bash
cd /home/docker_openvslam/catkin_ws/src/openvslam/build
mkdir slam localization
./run_tum_rgbd_slam_with_prior_map \
            -v /home/docker_openvslam/catkin_ws/src/openvslam/orb_vocab/orb_vocab.dbow2 \
            -d path/to/dataset/in/tum/format/06_mix_1_2020-03-17-14-47-54 \
            -c /home/docker_openvslam/catkin_ws/src/openvslam/config/husky_sdbcs_rgbd/Husky_SDBCS_03_17.yaml \
            --eval-log \
            --pmap-db path/to/vtk/map \
            --output_poses_slam slam/06_mix_1_2020-03-17-14-47-54.txt \
            --output_poses_loc localization/06_mix_1_2020-03-17-14-47-54.txt \
            --ref_poses path/to/gt/06_mix_1_2020-03-17-14-47-54.txt
```
При этом полученная траектория локализации будет сохранена в TUM формате в файл `localization/06_mix_1_2020-03-17-14-47-54.txt` в директории запуска, а траектория slam в файл `slam/06_mix_1_2020-03-17-14-47-54.txt`.
В файле `path/to/gt/06_mix_1_2020-03-17-14-47-54.txt` должна быть хотя бы одна строка с начальным положением камеры на карте в формате TUM.
Также для запуска необходимо указать путь к VTK файлу карты через параметр `--pmap-db`.

Если запуск осуществляется на Jetson, то его необходимо производить с правами суперпользователя (`sudo su`).

### Запуск ROS узлов

**RGBD SLAM**

launch файл - [ros/src/openvslam/launch/slam_rgbd.launch](../ros/src/openvslam/launch/slam_rgbd.launch)

В этом режиме метод принимает изображние и карту глубин и генерирует одометрию для base_link (фрейм odom связан c начальным положением base_link). Опционально публикует `tf: odom -> base_link`

**RGBD локализация**

launch файл - [ros/src/openvslam/launch/localization_rgbd_on_prior_map.launch](../ros/src/openvslam/launch/localization_rgbd_on_prior_map.launch)

В этом режиме метод принимает карту глубины, одометрию и опционально семантическую сегментацию. Генерирует локализацию для base_link (положение base_link на карте). Опционально публикует `tf: map -> odom`.

Также метод требует начальное положение на карте, передаваемое через топик `/initialpose`.

Если запуск осуществляется на Jetson, то его необходимо производить с правами суперпользователя (`sudo su`).

**RGBD SLAM + локализация**

launch файл - [ros/src/openvslam/launch/slam_localization_rgbd.launch](../ros/src/openvslam/launch/slam_localization_rgbd.launch)

Запускает `launch` файлы для SLAM и локализации с согласованными параметрами.

Если запуск осуществляется на Jetson, то его необходимо производить с правами суперпользователя (`sudo su`).
