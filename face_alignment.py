from __future__ import print_function
from torch import load
from skimage import io
from skimage import color

from models import FAN, ResNetDepth
from utils import *
from detection.sfd.sfd_detector import SFDDetector


class FaceAlignment:
    def __init__(self, device='cuda', flip_input=False, face_detector='sfd', verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.verbose = verbose

        network_size = 4

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        '''
        # Get the face detector
        face_detector_module = __import__('detection.' + face_detector,
                                          globals(), locals(), [face_detector], 0)
        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose)
        '''
        self.face_detector = SFDDetector(device=device, verbose=verbose)
        # Initialise the face alignemnt networks
        self.face_alignment_net = FAN(network_size)

        fan_weights = load('3DFAN.pth.tar', map_location=lambda storage, loc: storage)
        # Load all tensors onto GPU 1
        self.face_alignment_net.load_state_dict(fan_weights)

        self.face_alignment_net.to(device)
        self.face_alignment_net.eval()

        # Initialiase the depth prediciton network
        self.depth_prediciton_net = ResNetDepth()

        depth_weights = load('2D-to-3D.pth.tar', map_location=lambda storage, loc: storage)
        # Load all tensors onto GPU 1
        depth_dict = {
            k.replace('module.', ''): v for k,
                                            v in depth_weights['state_dict'].items()}
        self.depth_prediciton_net.load_state_dict(depth_dict)

        self.depth_prediciton_net.to(device)
        self.depth_prediciton_net.eval()

    def get_landmarks(self, image_or_path):
        tensor_or_path = torch.tensor(image_or_path)
        detected_faces = self.face_detector.detect_from_image(tensor_or_path)
        return self.get_landmarks_from_image(image_or_path, detected_faces)

    @torch.no_grad()
    def get_landmarks_from_image(self, image_or_path, detected_faces):
        """

        This function predicts a set of 68 3D images, one for each image present.
         Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image
        """
        if isinstance(image_or_path, str):
            try:
                image = io.imread(image_or_path)
            except IOError:
                print("error opening file :: ", image_or_path)
                return None
        elif isinstance(image_or_path, torch.Tensor):
            image = image_or_path.detach().cpu().numpy()
        else:
            image = image_or_path

        if image.ndim == 2:
            image = color.gray2rgb(image)
        elif image.ndim == 4:
            image = image[..., :3]

        if len(detected_faces) == 0:
            print("Warning: No faces were detected.")
            return None

        landmarks = []
        for i, d in enumerate(detected_faces):
            center = torch.FloatTensor(
                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
            center[1] = center[1] - (d[3] - d[1]) * 0.12
            scale = (d[2] - d[0] + d[3] - d[1]) / self.face_detector.reference_scale

            inp = crop(image, center, scale)
            inp = torch.from_numpy(inp.transpose(
                (2, 0, 1))).float()

            inp = inp.to(self.device)
            inp.div_(255.0).unsqueeze_(0)

            out = self.face_alignment_net(inp)[-1].detach()
            if self.flip_input:
                out += flip(self.face_alignment_net(flip(inp))
                            [-1].detach(), is_label=True)
            out = out.cpu()

            pts, pts_img = get_preds_fromhm(out, center, scale)
            pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)

            heatmaps = np.zeros((68, 256, 256), dtype=np.float32)
            for i in range(68):
                if pts[i, 0] > 0:
                    heatmaps[i] = draw_gaussian(
                        heatmaps[i], pts[i], 2)
            heatmaps = torch.from_numpy(
                heatmaps).unsqueeze_(0)

            heatmaps = heatmaps.to(self.device)
            depth_pred = self.depth_prediciton_net(
                torch.cat((inp, heatmaps), 1)).data.cpu().view(68, 1)
            pts_img = torch.cat(
                (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)

            landmarks.append(pts_img.numpy())
        return landmarks

    @staticmethod
    def remove_models(self):
        base_path = os.path.join(appdata_dir('face_alignment'), "data")
        for data_model in os.listdir(base_path):
            file_path = os.path.join(base_path, data_model)
            try:
                if os.path.isfile(file_path):
                    print('Removing ' + data_model + ' ...')
                    os.unlink(file_path)
            except Exception as e:
                print(e)
