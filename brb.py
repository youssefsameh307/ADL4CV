import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
def plot_3d_motion(joints,radius=4, plotName = 'newplot.png'):
#     matplotlib.use('Agg')
    title = ''
    max_len = joints.shape[0]
    target_frame = int(max_len//2)
    kinematic_tree = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    title_sp = title.split(' ')
    if len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    width_in_pixels = 480
    height_in_pixels = 480

    # Convert pixels to inches (1 inch = 2.54 cm)
    width_in_inches = width_in_pixels / plt.rcParams['figure.dpi']
    height_in_inches = height_in_pixels / plt.rcParams['figure.dpi']

    # Create the plot with the specified size
    # fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches))

    fig = plt.figure(figsize=(width_in_inches, height_in_inches))

    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['green', 'pink', 'blue', 'yellow', 'red',  
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
             'darkred', 'darkred','darkred','darkred','darkred']
    

    import random

    # Set a seed value (e.g., 1234 for reproducibility)
    random.seed(232113)

    num_colors = len(colors)
    colors = [(random.random(), random.random(), random.random()) for _ in range(num_colors+10)]


    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]
    
    # data[..., 0] -= data[:, 0:1, 0]
    # data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index , trajectory_only = 'False'):
        #         print(index)
        # ax.lines.clear()
        ax.collections.clear()
        # ax.lines = []
        # ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =

        # plot_xzPlane(MINS[0]
        #             #  -trajec[index, 0]
        #              , MAXS[0]
        #             #  -trajec[index, 0]
        #              ,0
        #              ,MINS[2]
        #             #    -trajec[index, 1]
        #                  ,MAXS[2]
        #                 #  -trajec[index, 1]
        #                  )


#         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)
        trajec_color = 'blue'
        if trajectory_only == 'True':
            trajec_color = 'red'

        if index > 1:
            ax.plot3D(trajec[:index, 0]
                    #   -trajec[index, 0] ## these center the trajectory
                      , np.zeros_like(trajec[:index, 0]),
                        trajec[:index, 1]
                        # -trajec[index, 1] ## these center the trajectory
                        , linewidth=1.0,
                      color=trajec_color
                      )
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])
        
        import matplotlib.cm as cm

        # cmap = cm.get_cmap('tab20')  # Choose a colormap
        # num_colors = 22  # Assuming kinematic_tree has a defined length

        cmap = cm.get_cmap('gist_ncar')
        num_colors = 26
        colors = cmap(np.linspace(0, 1, num_colors))
        if (trajectory_only == 'False'):
            
            for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
    #             print(color)
                # print(chain)
                if i < 5:
                    linewidth = 4.0
                else:
                    linewidth = 2.0

                for j in range(len(chain) - 1):
                    start_idx = chain[j]
                    
                    norm = start_idx / (num_colors - 1)
                    color = cmap(norm)

                    end_idx = chain[j + 1]
                    # color = colors[start_idx]
                    print(data[index])
                    ax.plot3D(
                        [data[index, start_idx, 0], data[index, end_idx, 0]],
                        [data[index, start_idx, 1], data[index, end_idx, 1]],
                        [data[index, start_idx, 2], data[index, end_idx, 2]],
                        linewidth=linewidth,
                        color=colors[start_idx]
            )

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_facecolor('black')
        
        # Set the color of the axes labels and ticks to white
        ax.w_xaxis.set_pane_color((0, 0, 0, 1))
        ax.w_yaxis.set_pane_color((0, 0, 0, 1))
        ax.w_zaxis.set_pane_color((0, 0, 0, 1))
        
        ax.xaxis._axinfo['grid'].update(color = 'w', linewidth = 0.5)
        ax.yaxis._axinfo['grid'].update(color = 'w', linewidth = 0.5)
        ax.zaxis._axinfo['grid'].update(color = 'w', linewidth = 0.5)
        
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')
        
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')
        
        fig.patch.set_facecolor('black')

    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)

    # ani.save(save_path, fps=fps)
    update(max_len,'True')
    update(target_frame)
    
    file_path = './conditions/' + plotName

    # Save the plot as a PNG file
    # buf = BytesIO()
    plt.savefig(file_path)
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # image = plt.imread(buf)
    # plt.show()
    # plt.close()
    # return image
    return ''
