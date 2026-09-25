from pathlib import Path

import betterosi
import polars as pl
import pyarrow.parquet as pq
import pytest

import omega_prime
from omega_prime.schemas import recording_moving_object_schema

TRUCK_IDX = 0
TRAILER_IDX = 1


def _mv(idx, subtype, has_trailer=False, trailer_id=None, x=0.0):
    return betterosi.MovingObject(
        id=betterosi.Identifier(value=idx),
        type=betterosi.MovingObjectType.TYPE_VEHICLE,
        base=betterosi.BaseMoving(
            dimension=betterosi.Dimension3D(length=5.0, width=2.0, height=2.0),
            position=betterosi.Vector3D(x=x, y=0.0, z=0.0),
            orientation=betterosi.Orientation3D(roll=0.0, pitch=0.0, yaw=0.0),
            velocity=betterosi.Vector3D(x=1.0, y=0.0, z=0.0),
            acceleration=betterosi.Vector3D(x=0.0, y=0.0, z=0.0),
        ),
        vehicle_classification=betterosi.MovingObjectVehicleClassification(
            type=subtype,
            role=betterosi.MovingObjectVehicleClassificationRole.ROLE_CIVIL,
            has_trailer=has_trailer,
            trailer_id=None if trailer_id is None else betterosi.Identifier(value=trailer_id),
        ),
    )


@pytest.fixture
def gts():
    return [
        betterosi.GroundTruth(
            version=betterosi.InterfaceVersion(version_major=3, version_minor=7, version_patch=9),
            timestamp=betterosi.Timestamp(seconds=0, nanos=i * 100_000_000),
            moving_object=[
                _mv(
                    TRUCK_IDX,
                    betterosi.MovingObjectVehicleClassificationType.TYPE_SEMITRACTOR,
                    has_trailer=True,
                    trailer_id=TRAILER_IDX,
                    x=float(i),
                ),
                _mv(TRAILER_IDX, betterosi.MovingObjectVehicleClassificationType.TYPE_SEMITRAILER, x=float(i) - 6),
            ],
        )
        for i in range(3)
    ]


def _assert_trailer_attributes(rec: omega_prime.Recording):
    truck = rec.df.filter(pl.col("idx") == TRUCK_IDX)
    trailer = rec.df.filter(pl.col("idx") == TRAILER_IDX)
    assert truck["has_trailer"].all()
    assert (truck["trailer_id"] == TRAILER_IDX).all()
    assert not trailer["has_trailer"].any()
    assert (trailer["trailer_id"] == -1).all()
    assert rec.moving_objects[TRUCK_IDX].has_trailer
    assert rec.moving_objects[TRUCK_IDX].trailer_id == TRAILER_IDX


def test_from_osi_gts(gts):
    rec = omega_prime.Recording.from_osi_gts(gts, validate=True)
    _assert_trailer_attributes(rec)


def test_to_osi_gts(gts):
    rec = omega_prime.Recording.from_osi_gts(gts)
    for gt in rec.to_osi_gts():
        mvs = {mv.id.value: mv for mv in gt.moving_object}
        assert mvs[TRUCK_IDX].vehicle_classification.has_trailer
        assert mvs[TRUCK_IDX].vehicle_classification.trailer_id.value == TRAILER_IDX
        assert not mvs[TRAILER_IDX].vehicle_classification.has_trailer
        assert mvs[TRAILER_IDX].vehicle_classification.trailer_id is None


@pytest.mark.parametrize("suffix", [".mcap", ".parquet"])
def test_file_roundtrip(gts, tmp_path: Path, suffix):
    path = tmp_path / f"rec{suffix}"
    rec = omega_prime.Recording.from_osi_gts(gts)
    if suffix == ".mcap":
        rec.to_mcap(path)
    else:
        rec.to_parquet(path)
    _assert_trailer_attributes(omega_prime.Recording.from_file(path, validate=True))


def test_interpolate(gts):
    rec = omega_prime.Recording.from_osi_gts(gts)
    rec.interpolate(hz=20)
    _assert_trailer_attributes(rec)


def test_missing_columns_are_filled(gts):
    rec = omega_prime.Recording.from_osi_gts(gts)
    df = rec.df.drop("has_trailer", "trailer_id", "frame", "polygon", "coords", "geometry")
    rec = omega_prime.Recording(df, validate=True)
    assert not rec.df["has_trailer"].any()
    assert (rec.df["trailer_id"] == -1).all()
    assert not rec.moving_objects[TRUCK_IDX].has_trailer
    for gt in rec.to_osi_gts():
        for mv in gt.moving_object:
            assert not mv.vehicle_classification.has_trailer
            assert mv.vehicle_classification.trailer_id is None


def test_parquet_without_trailer_columns(gts, tmp_path: Path):
    path = tmp_path / "rec.parquet"
    omega_prime.Recording.from_osi_gts(gts).to_parquet(path)
    pq.write_table(pq.read_table(path).drop_columns(["has_trailer", "trailer_id"]), path)
    rec = omega_prime.Recording.from_file(path, validate=True)
    assert not rec.moving_objects[TRUCK_IDX].has_trailer
    assert len(list(rec.to_osi_gts())) == len(gts)


def test_trailer_id_without_has_trailer_is_invalid(gts):
    rec = omega_prime.Recording.from_osi_gts(gts)
    df = rec.df.drop("polygon", "coords", "geometry").with_columns(has_trailer=pl.lit(False))
    with pytest.raises(Exception, match="trailer_id"):
        recording_moving_object_schema.validate(df, lazy=True)
