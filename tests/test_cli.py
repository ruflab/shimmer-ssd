from os import listdir
from shutil import rmtree

from click.testing import CliRunner

from simple_shapes_dataset.cli.alignments import add_alignment_split
from simple_shapes_dataset.cli.create_dataset import create_dataset
from simple_shapes_dataset.dataset.domain_alignment import get_alignment


def test_create_dataset(tmp_path):
    data_path = tmp_path / "_test_data"
    splits_path = data_path / "domain_splits_v2"
    runner = CliRunner()
    result_create = runner.invoke(
        create_dataset,
        [
            "-o",
            str(data_path.resolve()),
            "--ntrain",
            "2",
            "--nval",
            "1",
            "--ntest",
            "1",
        ],
    )
    assert result_create.exit_code == 0
    for k in range(2):
        assert (data_path / "train" / f"{k}.png").exists()

    result_alignment = runner.invoke(
        add_alignment_split,
        ["-p", str(data_path.resolve()), "-a", "v", "1.0", "-a", "t", "0.5"],
    )
    assert result_alignment.exit_code == 0
    assert len(listdir(splits_path)) == 3

    split = get_alignment(
        data_path, "train", {frozenset(["v"]): 1.0, frozenset(["t"]): 0.5}, 0, None
    )
    assert split[frozenset(["v"])].shape[0] == 2
    assert split[frozenset(["t"])].shape[0] == 1

    rmtree(data_path)
