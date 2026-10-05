"""Command-line inputs shared by the edit, texture and eval commands: the edited view's 2D mask and camera."""

from . import camera as cameras

CAMERA_FLAGS = ("camera_meta", "camera", "orbit")


def add_view_args(group) -> None:
    group.add_argument("--mask", type=str, help="2D mask of the edited view (single-channel PNG)")
    group.add_argument("--camera-meta", type=str, help="meta.json holding the edited view's camera")
    group.add_argument("--view", type=int, help="camera index in --camera-meta")
    group.add_argument("--camera", type=str, help="single-camera JSON")
    group.add_argument("--orbit", type=float, nargs=2, metavar=("ELEVATION", "AZIMUTH"),
                       help="benchmark-style orbit camera, degrees")


def given(args, names) -> list:
    return [f"--{name.replace('_', '-')}" if name != "whole" else "--global" for name in names
            if getattr(args, name, None) not in (None, False)]


def resolve_camera(args, size: int) -> cameras.Camera:
    chosen = given(args, CAMERA_FLAGS)
    if len(chosen) != 1:
        raise SystemExit(f"the edited view's camera needs exactly one of --camera-meta, --camera, --orbit; "
                         f"got {chosen}")
    if args.camera_meta is not None:
        if args.view is None:
            raise SystemExit("--camera-meta needs --view")
        camera = cameras.from_meta(args.camera_meta, args.view)
    elif args.camera is not None:
        camera = cameras.from_json(args.camera)
    else:
        camera = cameras.orbit(args.orbit[0], args.orbit[1], size)
    if camera.size != size:
        raise SystemExit(f"the camera frame is {camera.size} px but the edited view is {size} px")
    return camera
