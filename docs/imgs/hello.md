| Dataset | Radar tensor | Radar point cloud | Lidar point cloud | Camera | GPS | Bbox label | Tr. ID | Odom. | Weather conditions | Time | Num. labelled data | Num. labelled train data | Num. labelled val. data | Num. labelled test data | Num. Radar data | Num. Lidar data | Num. camera data | Num. 3D bboxes | Num. 2D bboxes | Num. points of objects | Road type | Driving period [hour] | Maximum range of Radar [m] |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| K-Radar (ours) | 4D | 4D | HR. | 360. | RTK | 3D | O | O | overcast, fog, rain, sleet, snow | d/n | 35K | 17.5K | - | 17.5K | 38.9K | 37.7K | 112K | 93K | - | - | urban, highway, alleyway, suburban, university, mountain, parking lots, shoulder | 1 | 118 |
| VoD | X | 4D | HR. | Front | RTK | 3D | O | O | X | day | 8.7K | 5.1K | 1.3K | 2.3K | n/a | n/a | n/a | 123K | - | - | urban | 0.2 | 64 |
| Astyx | X | 4D | LR. | Front | X | 3D | X | X | X | day | 0.5K | 0.4K | - | 1.3K | n/a | n/a | n/a | 3K | - | - | urban | 0.01 | 100 |
| RADDet | 3D | 3D | X | Front | X | 2D |  |  | X |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Zendar | 3D | 3D | LR. | Front | GPS | 2D |  |  | X |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RADIATE | 3D | 3D | LR. | Front | GPS | 2D |  |  | overcast, fog rain, snow |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| CARRADA | 3D | 3D | X | Front | X | 2D |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| CRUW | 3D | 3D | X | Front | X | Point |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| NuScenes | X | 3D | LR. | 360. | RTK | 3D |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Waymo | X | X | HR. | 360. | X | 3D |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| KITTI | X | X | HR. | Front | RTK | 3D |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| BDD100k | X | X | X | Front | RTK | 2D |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
