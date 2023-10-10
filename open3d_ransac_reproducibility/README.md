# Open3D RANSAC reproducibility

Issue: calling several times Open3D's `segment_plane` method returns different results, which impacts on the reproducibility of the experiments. See more info in [this close issue](https://github.com/isl-org/Open3D/issues/5647).

Outputs for the `plane_test.py` script for different Open3D versions:

(tag v0.16.0)

```bash
open3d version: 0.16.0+4eef4b3
Overlapping of inliers [100.0, 91.76458097395243, 89.25184031710079, 89.25184031710079, 89.25184031710079, 89.25184031710079, 89.25184031710079, 89.25184031710079, 89.25184031710079, 89.25184031710079]
Overlapping of inliers [True, False, False, False, False, False, False, False, False, False]
```

(tag v0.17.0)

```bash
open3d version: 0.17.0+9238339
Overlapping of inliers [100.0, 100.0, 96.72631432931058, 96.72631432931058, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
Overlapping of inliers [True, True, False, False, True, True, True, True, True, True]
```

(master)

```bash
open3d version: 0.17.0+ce442ea
Overlapping of inliers [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
Overlapping of inliers [True, True, True, True, True, True, True, True, True, True]
```
