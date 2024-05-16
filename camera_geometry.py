import numpy as np

def get_intrinsic_matrix_fov(field_of_view_deg, image_width_pix, image_height_pix):
    field_of_view_rad = field_of_view_deg * np.pi/180
    alpha_pix = (image_width_pix / 2.0) / np.tan(field_of_view_rad / 2.)
    return get_intrinsic_matrix(alpha_pix, image_width_pix, image_height_pix)

def get_intrinsic_matrix(focal_distance_pix, image_width_pix, image_height_pix):
    Cu = image_width_pix / 2.0
    Cv = image_height_pix / 2.0
    return np.array([[focal_distance_pix, 0                 , Cu            ],
                     [0                 , focal_distance_pix, Cv            ],
                     [0                 , 0                 , 1.0           ]])

def project_polyline(polyline_world, trafo_world_to_cam, K):
    x,y,z = polyline_world[:,0], polyline_world[:,1], polyline_world[:,2]
    homvec = np.stack((x,y,z,np.ones_like(x)))
    proj_mat = K @ trafo_world_to_cam[:3,:]
    pl_uv_cam = (proj_mat @ homvec).T
    u = pl_uv_cam[:,0] / pl_uv_cam[:,2]
    v = pl_uv_cam[:,1] / pl_uv_cam[:,2]
    return np.stack((u,v)).T

# https://github.com/thomasfermi/Algorithms-for-Automated-Driving
class CameraGeometry(object):
    def __init__(self, height_m=1.3, yaw_deg=0, pitch_deg=-5, roll_deg=0, image_width_pix=1024, image_height_pix=512, field_of_view_deg=45, focal_length_pix=None):
        # scalar constants
        self.height_m = height_m
        self.pitch_deg = pitch_deg
        self.roll_deg = roll_deg
        self.yaw_deg = yaw_deg
        self.image_width_pix = image_width_pix
        self.image_height_pix = image_height_pix
        self.field_of_view_deg = field_of_view_deg
        self.focal_length_pix = focal_length_pix
        # camera intriniscs and extrinsics
        if self.focal_length_pix:
            self.intrinsic_matrix = get_intrinsic_matrix(self.focal_length_pix,  self.image_width_pix, self.image_height_pix)
        else:
            self.intrinsic_matrix = get_intrinsic_matrix_fov(self.field_of_view_deg,  self.image_width_pix, self.image_height_pix)
        self.inverse_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)
        ## Note that "rotation_cam_to_road" has the math symbol R_{rc} in the book
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        roll = np.deg2rad(roll_deg)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        rotation_road_to_cam = np.array([[cr*cy+sp*sr+sy, cr*sp*sy-cy*sr, -cp*sy],
                                            [cp*sr, cp*cr, sp],
                                            [cr*sy-cy*sp*sr, -cr*cy*sp -sr*sy, cp*cy]])
        self.rotation_cam_to_road = rotation_road_to_cam.T # for rotation matrices, taking the transpose is the same as inversion
        self.translation_cam_to_road = np.array([0,-self.height_m,0])
        self.trafo_cam_to_road = np.eye(4)
        self.trafo_cam_to_road[0:3,0:3] = self.rotation_cam_to_road
        self.trafo_cam_to_road[0:3,3] = self.translation_cam_to_road
        # compute vector nc. Note that R_{rc}^T = R_{cr}
        self.road_normal_camframe = self.rotation_cam_to_road.T @ np.array([0,1,0])


    def camframe_to_roadframe(self,vec_in_cam_frame):
        return self.rotation_cam_to_road @ vec_in_cam_frame + self.translation_cam_to_road

    def uv_to_roadXYZ_camframe(self,u,v):
        # NOTE: The results depend very much on the pitch angle (0.5 degree error yields bad result)
        # Here is a paper on vehicle pitch estimation:
        # https://refubium.fu-berlin.de/handle/fub188/26792
        uv_hom = np.array([u,v,1])
        Kinv_uv_hom = self.inverse_intrinsic_matrix @ uv_hom
        denominator = self.road_normal_camframe.dot(Kinv_uv_hom)
        return self.height_m*Kinv_uv_hom/denominator
    
    def uv_to_roadXYZ_roadframe(self,u,v):
        r_camframe = self.uv_to_roadXYZ_camframe(u,v)
        return self.camframe_to_roadframe(r_camframe)

    def uv_to_roadXYZ_roadframe_iso8855(self,u,v):
        X,Y,Z = self.uv_to_roadXYZ_roadframe(u,v)
        return np.array([Z,-X,-Y]) # read book section on coordinate systems to understand this

    def precompute_grid(self,dist=60):
        cut_v = int(self.compute_minimum_v(dist=dist)+1)
        xy = []
        for v in range(cut_v, self.image_height_pix):
            for u in range(self.image_width_pix):
                X,Y,Z= self.uv_to_roadXYZ_roadframe_iso8855(u,v)
                xy.append(np.array([X,Y]))
        xy = np.array(xy)
        return cut_v, xy

    def compute_minimum_v(self, dist):
        """
        Find cut_v such that pixels with v<cut_v are irrelevant for polynomial fitting.
        Everything that is further than `dist` along the road is considered irrelevant.
        """        
        trafo_road_to_cam = np.linalg.inv(self.trafo_cam_to_road)
        point_far_away_on_road = trafo_road_to_cam @ np.array([0,0,dist,1])
        uv_vec = self.intrinsic_matrix @ point_far_away_on_road[:3]
        uv_vec /= uv_vec[2]
        cut_v = uv_vec[1]
        return cut_v