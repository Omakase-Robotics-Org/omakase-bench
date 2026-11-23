# D1 robot daemon API (subset)

`openapi.json` is a subset of the OpenAPI 3.1 document the D1 robot daemon publishes for its HTTP
API: the operations the bench speaks (listed under `x-vendored.operations`) and every component they
reference, copied unchanged by `backends/real/d1/vendor_api.py`. Source: `openapi/d1-firmwared.v1.json` of
the D1 daemon repository at commit `0b935623ddfca6a948cb491b75512992cee2f41b` (2026-11-18).

The bench validates every robot-level request against this file (`omakase_bench.harness.command_api`), and
`tests/backends/real/d1/test_api.py` fails when the bench's requests and this document drift apart. To update it,
run the tool on a newer copy of the document and review the diff.

The subset holds two kinds of operation: the commands a model's command lines and the real backend send
(arm mode, joint moves and emergency stop, end effector set and stop, lift, neck, chassis velocity, limits), and the state the real
backend reads to report measured joint positions (`GET /v1/arm/feedback`, `GET /v1/end_effectors`,
`GET /v1/slider/state`, `GET /v1/neck/state`, `GET /v1/chassis/amr/localization`), and the arm lease it holds
for a run (`GET`, `POST` and `DELETE /v1/arm/lease`). The localization operation's answer is the in-house
base daemon's and is not restated in the document; the bench reads its odometry pose. `x-vendored.schemas` lists the two schemas copied without an
operation referring to them: `WsRequest` and `WsResponse`, the frames of the WebSocket frontend, over which
the real backend streams joint targets.

Licence: this subset is distributed under the repository's Apache License 2.0 (`NOTICE`). Its
`info.license` field is the source repository's and stays only because the file is copied unchanged.

Because the file is copied unchanged, its descriptions name parts of the robot's software that the bench
does not ship:

- the daemon's name `d1-firmwared` (`info.description`; also the source file name
  `openapi/d1-firmwared.v1.json` in `x-vendored.source`, `backends/real/d1/vendor_api.py` and this file);
- the platform vocabulary extensions `x-omakase-auth` (on every operation), `info.x-omakase-kit` and
  `info.x-omakase-planes`;
- the shared-memory contracts `arm_command.shm/1`, `arm_feedback.shm/1`, `arm_status.shm/1` and
  `arm_events.shm/1`;
- in the state operations: the arm controller's record (`ArmFeedbackRecord`, its 1428-byte raw form, the
  controller's state codes and units), the lift drive's registers and alarm codes (`Pr0.15`, `Pr8.00`,
  `Er153`, register addresses), and the end-effector drivers' product and bus vocabulary;
- the WebSocket frontend and its default address `ws://127.0.0.1:4751/v1/ws`, the `x-ws-method` names and
  the configuration keys `[rest] cors_allow_origins` and `[ws] arm_feedback_max_hz`;
- operations outside the subset that `info.description` mentions: `soft_kill`, `release_soft_kill`, `clear_errors`, `recover`, `check_pose`,
  `brake_engage`, `pause_nav`, `stop_nav`, `/v1/chassis/maps/{scene}/image`,
  `/v1/chassis/ultrasonic/stream`, `/v1/recorder/dumps/{id}/files/{name}`, `/v1/state`;
- "the web console" and "a teleop station" as consumers of the daemon, and the robot's mass (110 kg);
- in operation and schema descriptions: "the kit" (the motion library whose HOME and STOW poses the daemon
  embeds), teleoperation as a consumer that streams joint targets at up to 200 Hz, the lease's motion
  `fence`, and the chassis error kinds of `ErrorKind` (`remote_control_mode`, `ros_unmapped`,
  `hand_drive_blocked`, `update_in_progress` and others).

`info.description` also states that the daemon is unauthenticated and bound to the loopback interface.
