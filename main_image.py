import face_alignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections


try:
    input_img = io.imread('/home/arseniy/Project/coursework/shortparis.jpg')
except FileNotFoundError:
    input_img = io.imread('/home/arseniy/Project/coursework/shortparis.jpg')


fa = face_alignment.FaceAlignment(device='cpu', flip_input=True)
preds = fa.get_landmarks(input_img)


plot_style = dict(marker='o',
                  markersize=4,
                  linestyle='-',
                  lw=2)

pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
              'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
              'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
              'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
              'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
              'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
              'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
              'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
              'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
              }

fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 2, 1)
ax.imshow(input_img)
if preds:
    for i in range(len(preds)):
        for pred_type in pred_types.values():
            ax.plot(preds[i][pred_type.slice, 0],
                    preds[i][pred_type.slice, 1],
                    color=pred_type.color, **plot_style)

ax.axis('off')

# 3D-Plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
if preds:
    for i in range(len(preds)):
        surf = ax.scatter(preds[i][:, 0] * 1.2,
                      preds[i][:, 1],
                      preds[i][:, 2],
                      c='cyan',
                      alpha=1.0,
                      edgecolor='b')
        for pred_type in pred_types.values():
            ax.plot3D(preds[i][pred_type.slice, 0] * 1.2,
                    preds[i][pred_type.slice, 1],
                    preds[i][pred_type.slice, 2], color='blue')

ax.view_init(elev=90., azim=90.)
ax.set_xlim(ax.get_xlim()[::-1])
plt.show()
