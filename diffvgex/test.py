import pydiffvg
import torch
import skimage
import numpy as np

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width = 480
canvas_height = 480
circle = pydiffvg.Circle(radius = torch.tensor(40.0),
                         center = torch.tensor([128.0, 128.0]))

circle2 = pydiffvg.Circle(radius = torch.tensor(30.0),
                         center = torch.tensor([100.0, 100.0]))


num_control_points = torch.tensor([0])

points = torch.tensor([[30.0,  30.0], # base
                    #    [150.0,  60.0], # control point
                    #    [ 90.0, 198.0], # control point
                       [ 100.0, 100.0]]) # base

path = pydiffvg.Path(num_control_points = num_control_points,
                     points = points,
                     is_closed = False,
                     stroke_width = torch.tensor(5.0))


points = torch.tensor([[100.123,  50.123], # base
                    #    [150.0,  60.0], # control point
                    #    [ 90.0, 198.0], # control point
                       [ 100.0, 100.0]]) # base

path2 = pydiffvg.Path(num_control_points = num_control_points,
                     points = points,
                     is_closed = False,
                     stroke_width = torch.tensor(5.0))

# path1 = pydiffvg.Path(num_control_points=torch.tensor(0.0),points=torch.tensor([[50.0, 50.0],[100.0, 100.0]]),stroke_width=torch.tensor(5.0),is_closed=False)


shapes = [circle,circle2,path
          ,path2
          ]

# using the same shape group will result in a difference between shapes
 
circle_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
    fill_color = torch.tensor([0.3, 0.3, 0.3, 1.0]))

circle_group2 = pydiffvg.ShapeGroup(shape_ids = torch.tensor([1]),
    fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]))

path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([2
                                                           ,3
                                                           ]),
                                 fill_color = None,
                                 stroke_color = torch.tensor([0.6, 0.3, 0.6, 0.8]))

shape_groups = [circle_group,circle_group2,path_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)

render = pydiffvg.RenderFunction.apply
img = render(480, # width
             480, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None,
             *scene_args)
# The output image is in linear RGB space. Do Gamma correction before saving the image.
pydiffvg.imwrite(img.cpu(), 'results/single_circle/target.png', gamma=2.2)
target = img.clone()