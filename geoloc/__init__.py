"""
geoloc — camera-agnostic pixel → GPS geolocation for fire contours.

    camera.py      CameraConfig: intrinsics, distortion, metadata field names
    metadata.py    FrameMetadata + MetadataLoader implementations (EXIF/XMP, sidecar CSV)
    projection.py  Pure geometry: pixel → ray → flat ground → ENU meters → lat/lon

No rendering or overlay code lives in this package.
"""
