from utils import PROJECT_DIR

from simple_shapes_dataset.dataset.domain import (
    Attribute,
    RawText,
    SimpleShapesAttributes,
    SimpleShapesImages,
    SimpleShapesRawText,
    SimpleShapesText,
    Text,
)


def test_image_domain():
    train_images = SimpleShapesImages(PROJECT_DIR / "sample_dataset", "train")

    assert len(train_images) == 4

    image = train_images[0]
    assert image.size == (32, 32)


def test_attribute_domain():
    train_attributes = SimpleShapesAttributes(PROJECT_DIR / "sample_dataset", "train")

    assert len(train_attributes) == 4

    attr = train_attributes[0]
    assert isinstance(attr, Attribute)


def test_raw_text_domain():
    train_text = SimpleShapesRawText(PROJECT_DIR / "sample_dataset", "train")

    assert len(train_text) == 4

    text = train_text[0]
    assert isinstance(text, RawText)


def test_text_domain():
    train_text = SimpleShapesText(PROJECT_DIR / "sample_dataset", "train")

    assert len(train_text) == 4

    item = train_text[0]
    assert isinstance(item, Text)
