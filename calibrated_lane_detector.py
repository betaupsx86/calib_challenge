import numpy as np
from lane_detector import LaneDetector
from camera_geometry import CameraGeometry

def find_dist_to_line(point,line):
    x1,y1,x2,y2 = line
    x0,y0 = point
    ab = np.array([x2,y2]) - np.array([x1,y1])
    ap = np.array([x0,y0]) - np.array([x1,y1])
    distance = np.cross(ab,ap)/np.linalg.norm(ab)
    return distance

def find_dist_to_point(point,point_d):
    x1,y1 = point_d
    x0,y0 = point
    ab = np.array([x1,y1]) - np.array([x0,y0])
    return np.linalg.norm(ab)

# least square intersect of many lines
def intersect(P0,P1):
    """P0 and P1 are NxD arrays defining N lines.
    D is the dimension of the space. This function 
    returns the least squares intersection of the N
    lines from the system given by eq. 13 in 
    http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf.
    """
    n = (P1-P0)/np.linalg.norm(P1-P0,axis=1)[:,np.newaxis] # normalized
    projs = np.eye(n.shape[1]) - n[:,:,np.newaxis]*n[:,np.newaxis]  # I - n*n.T
    R = projs.sum(axis=0)
    q = (projs @ P0[:,:,np.newaxis]).sum(axis=0)
    p = np.linalg.lstsq(R,q,rcond=None)[0]
    return p

def get_intersection(line1, line2):
    m1, c1 = line1
    m2, c2 = line2
    if m1 == m2:
        return None
    u_i = (c2 - c1) / (m1 - m2)
    v_i = m1*u_i + c1
    return u_i, v_i

def get_py_from_vp(u_i, v_i, K):
    p_infinity = np.array([u_i, v_i, 1])
    K_inv = np.linalg.inv(K)
    r3 = K_inv @ p_infinity    
    r3 /= np.linalg.norm(r3)
    yaw = -np.arctan2(r3[0], r3[2])
    pitch = np.arcsin(r3[1])    
    return pitch, yaw

def get_vp_from_py(pitch, yaw, K):
    r3 = np.vstack([-np.cos(-yaw)*np.sin(-pitch), np.sin(-yaw), np.cos(-yaw)*np.cos(-pitch)])
    uv = K.dot(r3).squeeze()
    return int(uv[0]), int(uv[1])

# Camera extrinsics autocalibration based on vanishing point estimation.
# Neural net based lane detection ended up being the most reliable way to get the relevant vanishing point 
# Also tried the following :
# Traditional edge detectors with RANSAC
# Neural nets to directly estimate the vanishing point

# Adapted from https://thomasfermi.github.io/Algorithms-for-Automated-Driving/CameraCalibration/VanishingPointCameraCalibration.html
# https://github.com/thomasfermi/Algorithms-for-Automated-Driving
class CalibratedLaneDetector(LaneDetector):
    def __init__(self, config, checkpoint, device, pitch_yaw_history_window=10, cam_geom=CameraGeometry(), recalibrate=False, vp_from_all_lanes=False):
        """Lane detector that recalibrates the camera based on colected vanishing points.
        Args:
            config (str): Lane detector model config
            checkpoint (str): Lane detector model weights
            device (str): Lane detector model device
            pitch_yaw_history_window (int): Window size for pitch and yaw collection and averaging
            can_geom (CameraGeometry): Camera geometry
            recalibrate (bool): Whether to adjust the camera geometry
            vp_from_all_lanes (bool): Whether to use all detected lanes to determine the vanishing point
        """
        super().__init__(config, checkpoint, device, cam_geom)
        self.estimated_pitch_deg = 0
        self.estimated_yaw_deg = 0
        self.mean_residuals_thresh = 15
        self.slope_acceptance_thresh_lo = 0.25
        self.slope_acceptance_thresh_hi = 1.5
        self.update_cam_geometry()
        self.pitch_yaw_history = []
        self.pitch_yaw_history_window = pitch_yaw_history_window
        self.calibration_success = False
        self.vanishing_point = None
        self.vanishing_point_lane_mask = []
        self.recalibrate = recalibrate
        self.vp_from_all_lanes = vp_from_all_lanes


    def get_fit_and_probs(self, image):
        _, preds = self.detect(image)

        P0,P1=[],[]
        self.vanishing_point_lane_mask = [False]*len(preds)
        # Find a better way to filter curved lanes. Yaw estimation is very susceptible to this
        lines = [self._fit_line_v_of_u(pred) for pred in preds]
        line_left = None
        line_right = None
        line_left_id = None
        line_right_id = None
        line_left_m_abs = self.slope_acceptance_thresh_lo
        line_right_m_abs = self.slope_acceptance_thresh_lo
        for i,line in enumerate(lines):
            if line is None:
                continue
            m = line.c[0]
            if abs(m) <= self.slope_acceptance_thresh_hi and abs(m) >= self.slope_acceptance_thresh_lo:
                self.vanishing_point_lane_mask[i] = True
                P0.append([0, line(0)])
                P1.append([1, line(1)])
            if m < 0 and abs(m) <= self.slope_acceptance_thresh_hi and abs(m) >= line_left_m_abs:
                line_left = line
                line_left_id = i
                line_left_m_abs = abs(m)
            elif m > 0 and abs(m) <= self.slope_acceptance_thresh_hi and abs(m) >= line_right_m_abs:
                line_right = line
                line_right_id = i
                line_right_m_abs = abs(m)

        vanishing_point = None
        if self.vp_from_all_lanes and len(P0) >= 2:
            vanishing_point = intersect(np.array(P0), np.array(P1))
        elif (line_left is not None) and (line_right is not None):
            self.vanishing_point_lane_mask = [False]*len(preds)
            self.vanishing_point_lane_mask[line_left_id] = True
            self.vanishing_point_lane_mask[line_right_id] = True
            vanishing_point = get_intersection(line_left, line_right)

        if vanishing_point is not None:
            self.vanishing_point = (int(vanishing_point[0]),int(vanishing_point[1]))
            pitch, yaw = get_py_from_vp(*self.vanishing_point, self.cg.intrinsic_matrix)
            self.add_to_pitch_yaw_history(pitch, yaw)

        fits = [self.fit_poly(pred) for pred in preds]
        return fits, preds
    
    def _fit_line_v_of_u(self, pred):
        if len(pred) == 0:
            return None
        coeffs, residuals, _, _, _ = np.polyfit(
            pred[:,0], pred[:,1], deg=1, full=True)
            
        mean_residuals = residuals/len(pred)
        if mean_residuals > self.mean_residuals_thresh:
            return None
        else:
            return np.poly1d(coeffs)

    def add_to_pitch_yaw_history(self, pitch, yaw):
        self.pitch_yaw_history.append([pitch, yaw])
        if len(self.pitch_yaw_history) > self.pitch_yaw_history_window:
            py = np.array(self.pitch_yaw_history)
            mean_pitch = np.mean(py[:,0])
            mean_yaw = np.mean(py[:,1])
            self.estimated_pitch_deg = np.rad2deg(mean_pitch)
            self.estimated_yaw_deg = np.rad2deg(mean_yaw)
            self.calibration_success = True
            self.pitch_yaw_history = []
            if self.recalibrate:
                self.update_cam_geometry()

    def update_cam_geometry(self):
        self.cg = CameraGeometry(
            height_m = self.cg.height_m, 
            roll_deg = self.cg.roll_deg,
            pitch_deg = self.estimated_pitch_deg, 
            yaw_deg = self.estimated_yaw_deg,
            field_of_view_deg = self.cg.field_of_view_deg,
            focal_length_pix=self.cg.focal_length_pix,
            image_width_pix = self.cg.image_width_pix,
            image_height_pix = self.cg.image_height_pix,
        )
        self.cut_v, self.grid = self.cg.precompute_grid()

    def get_estimated_pitch_yaw_rad(self):
        return np.deg2rad(self.estimated_pitch_deg), np.deg2rad(self.estimated_yaw_deg)

    def get_vanishing_point(self):
        return self.vanishing_point
    
    def get_vanishing_point_lane_mask(self):
        return self.vanishing_point_lane_mask
    
    def draw_lanes_and_vanishing_point(self, frame, preds):
        assert len(preds) == len(self.vanishing_point_lane_mask)
        colors = [(255, 0, 0) if l else (0, 255, 0) for l in self.vanishing_point_lane_mask]
        if self.vanishing_point:
            u_i, v_i = self.vanishing_point
            pred_vp = [np.array([[u_i-10,v_i], [u_i+10,v_i]]), np.array([[u_i, v_i+10], [u_i, v_i-10]])]
            return self.draw_lanes_colors(frame, preds + pred_vp, colors + [(255, 0, 0), (255, 0, 0)])
        else:      
            return self.draw_lanes_colors(frame, preds, colors)       
