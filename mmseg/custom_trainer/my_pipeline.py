import warnings 
warnings.filterwarnings('ignore')
import cv2

import numpy as np

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO

# 시각화를 위한 라이브러리
import seaborn as sns; sns.set()

import mmcv

from mmseg.datasets.builder import DATASETS, PIPELINES
from mmseg.datasets.custom import CustomDataset
from mmseg.core import eval_metrics, pre_eval_to_metrics
from mmseg.datasets.pipelines import Compose

import os.path as osp
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable


# segmentation map을 online으로 생성하기 위해서 __init__ 함수를 재정의합니다.

@PIPELINES.register_module()
class CustomLoadAnnotations(object):
    def __init__(self,
                 coco_json_path,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.coco = COCO(coco_json_path)
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        # load seg_map from coco
        coco_ind = results['ann_info']['coco_image_id']
        image_info = self.coco.loadImgs(coco_ind)[0]
        ann_inds = self.coco.getAnnIds(coco_ind)
        anns = self.coco.loadAnns(ann_inds)
        anns = list(sorted(anns, key=lambda x: -x['area']))
        
        gt_semantic_seg = np.zeros((image_info["height"], image_info["width"]))
        for ann in anns:
            gt_semantic_seg[self.coco.annToMask(ann) == 1] = ann['category_id']
        gt_semantic_seg = gt_semantic_seg.astype(np.int64)

        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

categories = [
    {
        "id": 1,
        "name": "General trash",
        "supercategory": "General trash"
    },
    {
        "id": 2,
        "name": "Paper",
        "supercategory": "Paper"
    },
    {
        "id": 3,
        "name": "Paper pack",
        "supercategory": "Paper pack"
    },
    {
        "id": 4,
        "name": "Metal",
        "supercategory": "Metal"
    },
    {
        "id": 5,
        "name": "Glass",
        "supercategory": "Glass"
    },
    {
        "id": 6,
        "name": "Plastic",
        "supercategory": "Plastic"
    },
    {
        "id": 7,
        "name": "Styrofoam",
        "supercategory": "Styrofoam"
    },
    {
        "id": 8,
        "name": "Plastic bag",
        "supercategory": "Plastic bag"
    },
    {
        "id": 9,
        "name": "Battery",
        "supercategory": "Battery"
    },
    {
        "id": 10,
        "name": "Clothing",
        "supercategory": "Clothing"
    }
]
category_names = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 
'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

categories
@DATASETS.register_module()
class TrashDataset(CustomDataset):
    
    # CLASSES = ['background'] + [cat['name'] for cat in categories]
    CLASSES = category_names
    PALETTE = None    # random generation
    
    def __init__(
        self,
        pipeline,
        coco_json_path, #################
        img_dir,
        is_valid, #################
        img_suffix='.jpg',
        ann_dir=None,
        seg_map_suffix='.png',
        split=None,
        data_root=None,
        test_mode=False,
        ignore_index=255,
        reduce_zero_label=False,
        classes=None,
        palette=None,
        gt_seg_map_loader_cfg=None,
        file_client_args=dict(backend='disk') #################
    ):
        
        ann_dir = None    # 필요가 없기 때문에 None으로...
        
        self.is_valid = is_valid
        self.coco = COCO(coco_json_path)
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = int(ignore_index)
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(classes, palette)
        
        # gt_seg_map_loader의 재정의가 필요합니다.
        self.gt_seg_map_loader = CustomLoadAnnotations(coco_json_path) #########################

        self.file_client_args = file_client_args
        self.file_client = mmcv.FileClient.infer_client(self.file_client_args)

        if test_mode:
            assert self.CLASSES is not None, '`cls.CLASSES` or `classes` should be specified when testing'

        
        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations()#########################
    
    def load_annotations(self): #########################
        img_infos = []
        
        for img in self.coco.imgs.values():
            img_info = dict(filename=img['file_name'])
            img_infos.append(img_info)
            
            img_info['ann'] = dict(coco_image_id=img['id'])
        
        return img_infos

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']
    
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        # results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map
    
    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)

            ### 256, 256으로 작게 만든다음에 평가 => evaluation 속도 빠르게 하기 위해서인듯?
            
            for i in range(len(results)):
                results[i] = cv2.resize(results[i], dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
            
            gt_seg_maps_resized = []
            for gt_seg_map in gt_seg_maps:
                gt_seg_maps_resized.append(
                    cv2.resize(gt_seg_map, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
                )
            
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps_resized,
                num_classes,
                self.ignore_index,
                metric,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        return eval_results
