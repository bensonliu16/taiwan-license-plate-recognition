# Model setup

Place the custom YOLO license-plate detector at:

```text
models/license_plate_yolo.pt
```

The original development folder contains two identical copies of the primary
model, `carplate_11m.pt` and `Carplate_best.pt`. Use either one locally and
rename the copy to `license_plate_yolo.pt`.

Model weights are intentionally excluded from Git because they are large binary
artifacts. Do not publish weights unless you own them and have confirmed that
the training dataset and model license allow redistribution.
