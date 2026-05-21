import numpy as np
from PIL import Image

import osam
import osam.apis
import osam.types


def run_sam2_by_point(
        image_path,
        x,
        y,
        output_path="mask.png",
        model_name="sam2"
):
    """
    model_name:
        sam2        = balanced
        sam2:large  = accuracy
    """

    image_pil = Image.open(image_path).convert("RGB")
    image = np.asarray(image_pil)

    request = osam.types.GenerateRequest(
        model=model_name,
        image=image,
        prompt=osam.types.Prompt(
            points=[[x, y]],
            point_labels=[1]  # 1 = 前景点
        ),
    )

    response = osam.apis.generate(request=request)

    if not response.annotations:
        raise RuntimeError("SAM2 没有生成任何 mask，请换一个点或者改用框选方式。")

    # 一般取第一个结果
    annotation = response.annotations[0]

    # annotation.mask 通常只是 bbox 区域内的小 mask
    bbox = annotation.bounding_box

    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    small_mask = np.asarray(annotation.mask)

    # 转成 0/255 图像
    if small_mask.dtype != np.bool_:
        small_mask = small_mask > 0

    full_mask[
        bbox.ymin:bbox.ymax + 1,
        bbox.xmin:bbox.xmax + 1
    ] = small_mask.astype(np.uint8) * 255

    Image.fromarray(full_mask).save(output_path)

    print(f"mask saved: {output_path}")


if __name__ == "__main__":
    run_sam2_by_point(
        image_path=r"D:\Desktop\现场余料图\B2309045.jpg",
        x=500,
        y=400,
        output_path="mask2.png",
        model_name="sam2"
    )