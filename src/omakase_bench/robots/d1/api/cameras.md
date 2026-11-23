# D1 cameras: what a client can read (interface description)

This file describes interfaces; it copies no source. It was written from the documentation and the
route tables of the D1's camera software at the commits named below.

## Summary

| Source | Where it is reachable | Format | Cameras | Depth | Used by the bench |
| --- | --- | --- | --- | --- | --- |
| Camera daemon, video sinks | on the robot only: V4L2 loopback devices `/dev/video100` .. `/dev/video107` | raw video frames | every output a graph gives a device | as a separate video output where configured | colour, `camera_source=v4l2` |
| Camera daemon, shared-memory topics | on the robot only: iceoryx2 topics under the daemon account's runtime directory, readable by that account | fixed-schema video frames (a 72-byte header, for depth a 136-byte block with the depth imager's intrinsics, then pixels), point clouds, metadata | every output a graph gives a topic | yes: gray16 depth video on the depth imager's own grid | colour, `camera_source=iceoryx`; depth |
| Camera daemon, control socket | on the robot only: a Unix-domain socket, JSON lines | requests and replies; a `capture` session writes frames of one node to a file | one node per session | no | no |
| Graph editor server | HTTP on port 8790, loopback by default, no authentication | preview as WebRTC H.264 media tracks, negotiated by `POST /rtc/offer`; no still image and no MJPEG over HTTP | the taps a session asks for | no | no |
| Calibration service | HTTP on `127.0.0.1:8795` (other addresses are refused), reached from another host through an SSH tunnel | `GET /frame.jpg`: one JPEG, `Content-Type: image/jpeg`, `Cache-Control: no-store`; `GET /stream.mjpg`: `multipart/x-mixed-replace; boundary=frame`, about 8 to 15 frames a second | the one camera the calibration step has selected | no | the snapshot format |
| Calibration file | on the robot: `~/.config/omakase/camera_calibration.json`, schema `omakase.camera_calibration/2` | per camera slot (head, left_wrist, right_wrist): intrinsics, distortion, mount pose, gates, provenance | all | n/a | yes: lens and mount for the depth lookup |

Commits read: camera daemon and editor `d621394` (release 0.6.0), calibration service `f2bcc28`
(release 0.8.0).

## Which output is which camera

The camera daemon's graph decides, per robot. What its repositories state for the D1, and what
`robots/d1/robot.yaml` `real.cameras` therefore holds as on-robot defaults:

| Camera | Video topic | Depth topic | Video device |
| --- | --- | --- | --- |
| head | `dxr/camera/head/raw` | `dxr/camera/head/depth` | `/dev/video100` (colour), `/dev/video101` (depth as video) |
| left_wrist | `dxr/camera/left_wrist/raw` | none | none |
| right_wrist | `dxr/camera/right_wrist/raw` | none | none |

The three video topics are those of the daemon's D1 profiles. The head's depth topic and devices are those
of the daemon's example graph for the D1's depth camera; the D1 profiles themselves give no camera a video
device, so on a robot running a profile unchanged only `camera_source=iceoryx` finds all three cameras, and
depth needs the depth nodes of the example graph added.

## What the real backend consumes

- On the robot (`camera_source=iceoryx` or `v4l2`): the video topics or devices above for colour, the depth
  topic for depth, the calibration file for lenses and mounts.
- From another host (`camera_source=http`): one URL per camera that answers `GET` with a single JPEG of the
  newest frame, `Content-Type: image/jpeg`: the format of the calibration service's `/frame.jpg`.

The backend has no default host and opens nothing unless it is enabled.

## What is missing

- The transform from the depth imager to the colour camera on the depth topic. The daemon reads the
  sensor's factory transform, but the depth frames are on the depth imager's grid and their depth block
  carries intrinsics only. The backend takes the transform from the robot description (`robot.yaml`
  `real.depth_to_colour`), which is zero for the D1: the two are taken as one frame, off by the sensor's
  baseline of about 15 mm on the head camera. The way round it is a graph that renders the point cloud into
  the colour view (`proc.cloud_to_video`, `view: color`), whose frames are on the colour camera's grid with
  the colour camera's intrinsics; the backend reads such a topic with `depth_frame: color`
  (`docs/real_robot.md`).
- Depth for the wrist cameras in the D1 graphs: the depth lookup covers the head camera only.
- For a remote client: a snapshot endpoint per camera, and depth and calibration over the network.
