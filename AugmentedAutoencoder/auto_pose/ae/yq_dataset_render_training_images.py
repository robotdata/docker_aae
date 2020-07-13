
import numpy as np
import functools

from .pysixd_stuff import transform


# https://danijar.com/structuring-your-tensorflow-models/
def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator



class Practice():
    def __init__(self):
        self.noof_training_imgs = 20000
        pass

    @lazy_property
    def renderer(self):
        from auto_pose.meshrenderer import meshrenderer, meshrenderer_phong
        renderer = meshrenderer.Renderer(
            "/path/to/my_3d_model.ply",
            1,
            "/home/yq/Data/23_pose_estimation/aae/autoencode_ws/tmp_datasets",
            
        )
        

    def render_training_images(self):
        H, W = 128, 128
        render_dims = (720, 540)
        K = [1075.65, 0, 720/2, 0, 1073.90, 540/2, 0, 0, 1]
        K = np.array(K).reshape(3,3)
        clip_near = 10
        clip_far = 10000
        pad_factor = 1.2
        max_rel_offset = 0.2
        t = np.array([0,0, 700.0])
        
        for i in np.arange(self.noof_training_imgs):
            R = transform.random_rotation_matrix()[:3,:3]
            bgr_x, depth_x = self.renderer.render(
                obj_id=0,
                W=render_dims[0],
                H=render_dims[1],
                K=K.copy(),
                R=R,
                t=t,
                near=clip_near,
                far=clip_far,
                random_light=True
            )
            