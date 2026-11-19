# Next recording brief

Hold a single world-space camera view for the whole repeated-drop sequence. Ball positions, bounces, joint configurations and shot numbers do not steer the camera. The view matches the perspective used for the appearance previews and includes the foreground board, gripper, bucket, nearby floor and tables behind it.

The next plan schedules twelve separately simulated foreground releases at row 1, column 2. Four previously recorded shot configurations repeat three times. Other tables execute the same concurrent batches. The exporter requires at least ten distinct measured foreground drop/rebound events and at least two active projected background drops in each shot before publishing the station videos. Rejected approaches and copied video frames cannot satisfy that count. All twelve planned chapters currently have to complete. Recording can resume at a verified camera cut after a failed approach.

Use `--recording-plan fixed-station-v1` with `film_cylinder_farm.py --dark-replay ...`. This is now the default for new replay commands. Old records without plan metadata continue to use `dark-tour-v1` during verification, and reproducing the historical tour requires that flag explicitly.

Apply the user's chosen gray/blue-gray tabletop, floor and soft lighting before the next capture. The appearance choice is still pending. The fixed-camera plan and export guards passed mock/frustum tests. Twelve actual drops and their encoded visibility will be checked in the next recording. This preparation did not launch a new capture.
