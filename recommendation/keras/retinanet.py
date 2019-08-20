import keras
from keras_retinanet.models.retinanet import AnchorParameters, __build_anchors
from keras_retinanet.models.retinanet import retinanet
from keras_retinanet import layers


# Forked to allow editing nms_threshold
def retinanet_bbox(
        model=None,
        anchor_parameters=AnchorParameters.default,
        nms=True,
        nms_threshold=0.5,
        class_specific_filter=True,
        name='retinanet-bbox',
        **kwargs
):
    """ Construct a RetinaNet model on top of a backbone and adds convenience functions to output boxes directly.

    This model uses the minimum retinanet model and appends a few layers to compute boxes within the graph.
    These layers include applying the regression values to the anchors and performing NMS.

    Args
        model                 : RetinaNet model to append bbox layers to. If None, it will create a RetinaNet model using **kwargs.
        anchor_parameters     : Struct containing configuration for anchor generation (sizes, strides, ratios, scales).
        nms                   : Whether to use non-maximum suppression for the filtering step.
        class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
        name                  : Name of the model.
        *kwargs               : Additional kwargs to pass to the minimal retinanet model.

    Returns
        A keras.models.Model which takes an image as input and outputs the detections on the image.

        The order is defined as follows:
        ```
        [
            boxes, scores, labels, other[0], other[1], ...
        ]
        ```
    """
    if model is None:
        model = retinanet(num_anchors=anchor_parameters.num_anchors(), **kwargs)

    # compute the anchors
    features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]
    anchors = __build_anchors(anchor_parameters, features)

    # we expect the anchors, regression and classification values as first output
    regression = model.outputs[0]
    classification = model.outputs[1]

    # "other" can be any additional output from custom submodels, by default this will be []
    other = model.outputs[2:]

    # apply predicted regression to anchors
    boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = layers.FilterDetections(
        nms=nms,
        nms_threshold=nms_threshold,
        class_specific_filter=class_specific_filter,
        name='filtered_detections'
    )([boxes, classification] + other)

    outputs = detections

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=outputs, name=name)
