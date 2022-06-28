import argparse
from PIL import ImageDraw, Image
from yolo.yolo import YOLO, detect_video, detect_img

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
#####################################################################
class yolo_args:
    pass

def get_args():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='model-weights/YOLO_Face.h5',
    #                     help='path to model weights file')
    # parser.add_argument('--anchors', type=str, default='cfg/yolo_anchors.txt',
    #                     help='path to anchor definitions')
    # parser.add_argument('--classes', type=str, default='cfg/face_classes.txt',
    #                     help='path to class definitions')
    # parser.add_argument('--score', type=float, default=0.5,
    #                     help='the score threshold')
    # parser.add_argument('--iou', type=float, default=0.45,
    #                     help='the iou threshold')
    # parser.add_argument('--img-size', type=list, action='store',
    #                     default=(416, 416), help='input image size')
    # parser.add_argument('--image', default="samples/outside_000001.jpg", action="store_true",
    #                     help='image detection mode')
    # parser.add_argument('--video', type=str, default='samples/subway.mp4',
    #                     help='path to the video')
    # parser.add_argument('--output', type=str, default='outputs/',
    #                     help='image/video output path')
    # args = parser.parse_args()
    args = yolo_args()
    args.model = "model-weights/YOLO_Face.h5"
    args.anchors = "cfg/yolo_anchors.txt"
    args.classes = "cfg/face_classes.txt"
    args.score = 0.5
    args.iou = 0.45
    args.img_size = (416,416)
    args.image = "samples/outside_000001.jpg"
    return args


def _main():
    # Get the arguments
    args = get_args()

    if args.image:
        # Image detection mode
        print('[i] ==> Image detection mode\n')
        yolo = YOLO(args)
        
        res_image, _= yolo.detect_image(Image.open(args.image))
        print(_)
        print(_.shape)
        res_image.show()
        yolo.close_session()
    else:
        print('[i] ==> Video detection mode\n')
        # Call the detect_video method here
        detect_video(YOLO(args), args.video, args.output)

    print('Well done!!!')


if __name__ == "__main__":
    _main()
