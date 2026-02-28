import bpy
import os

texture_folder = r"C:\Users\User\Desktop\StripeRecognition\.calibrated structured light\Stereo calibration\Greycode pattern"
output_folder  = r"C:\Users\User\Desktop\StripeRecognition\.calibrated structured light\Stereo calibration\Greycode images\Pose5"

light = bpy.data.objects["Light"]
node = light.data.node_tree.nodes["Image Texture"]

for i in range(1, 22):
    filename = f"{(i):02d}.jpg"
    img_path = os.path.join(texture_folder, filename)

    if os.path.exists(img_path):
        img = bpy.data.images.load(img_path, check_existing=True)
        node.image = img

        bpy.context.scene.render.filepath = os.path.join(
            output_folder, f"render_{i:02d}.png"
        )

        bpy.ops.render.render(write_still=True)

print("Fertig.")