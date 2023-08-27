import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
from torchvision.ops import box_convert
import torchvision.transforms as TS

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from segment_anything import sam_model_registry, SamPredictor
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model as dino_load_model, predict as dino_predict, annotate as dino_annotate

# rembg
from rembg import new_session, remove

# Tag2Text
import sys
sys.path.append('../Grounded-Segment-Anything')
sys.path.append('../Grounded-Segment-Anything/Tag2Text')
from Tag2Text.models import tag2text
from Tag2Text import inference_ram

# # BLIP
# import nltk
# from transformers import BlipProcessor, BlipForConditionalGeneration

models = {
	'vit_h': "/mnt/local2T_v2/yinglin/sam_vit_h_4b8939.pth",
    'vit_b': './pretrained/sam_vit_b_01ec64.pth'
}

vitmatte_models = {
	'vit_b': "/mnt/local2T_v2/yinglin/ViTMatte_B_DIS.pth",
}

vitmatte_config = {
	'vit_b': './configs/matte_anything.py',
}

grounding_dino = {
    'config': '../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
    'weight': "/mnt/local2T_v2/linyx/image-matting/GroundingDINO/weights/groundingdino_swint_ogc.pth"
}

tag2text_models = {
    'ram': "/mnt/local2T_v2/yinglin/ram_swin_large_14m.pth",
}

blip_model = {
    'weight': "/mnt/local2T_v2/linyx/image-matting/blip-image-captioning-large",
}

def generate_checkerboard_image(height, width, num_squares):
    num_squares_h = num_squares
    square_size_h = height // num_squares_h
    square_size_w = square_size_h
    num_squares_w = width // square_size_w
    

    new_height = num_squares_h * square_size_h
    new_width = num_squares_w * square_size_w
    image = np.zeros((new_height, new_width), dtype=np.uint8)

    for i in range(num_squares_h):
        for j in range(num_squares_w):
            start_x = j * square_size_w
            start_y = i * square_size_h
            color = 255 if (i + j) % 2 == 0 else 200
            image[start_y:start_y + square_size_h, start_x:start_x + square_size_w] = color

    image = cv2.resize(image, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image

def init_segment_anything(model_type):
    """
    Initialize the segmenting anything with model_type in ['vit_b', 'vit_l', 'vit_h']
    """
    
    sam = sam_model_registry[model_type](checkpoint=models[model_type]).to(device)
    predictor = SamPredictor(sam)

    return predictor

def init_vitmatte(model_type):
    """
    Initialize the vitmatte with model_type in ['vit_s', 'vit_b']
    """
    cfg = LazyConfig.load(vitmatte_config[model_type])
    vitmatte = instantiate(cfg.model)
    vitmatte.to(device)
    vitmatte.eval()
    DetectionCheckpointer(vitmatte).load(vitmatte_models[model_type])

    return vitmatte

def init_tag2text(model_type):
    ram_model = tag2text.ram(pretrained=tag2text_models[model_type], image_size=384, vit='swin_l')
    ram_model.eval()
    ram_model = ram_model.to(device)
    return ram_model

def init_blip(model_weight):
    processor = BlipProcessor.from_pretrained(model_weight)
    blip_model = BlipForConditionalGeneration.from_pretrained(model_weight, torch_dtype=torch.float16).to(device)

    lemma = nltk.wordnet.WordNetLemmatizer()
    nltk.download(['punkt', 'averaged_perceptron_tagger', 'wordnet'])

    return processor, blip_model, lemma

def generate_trimap(mask, erode_kernel_size=10, dilate_kernel_size=10):
    erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    eroded = cv2.erode(mask, erode_kernel, iterations=5)
    dilated = cv2.dilate(mask, dilate_kernel, iterations=5)
    trimap = np.zeros_like(mask)
    trimap[dilated==255] = 128
    trimap[eroded==255] = 255
    return trimap

# once user upload an image, the original image is stored in `original_image`
def store_img(img):
    return img  # when new image is uploaded, `selected_points` should be empty

def convert_pixels(gray_image, boxes):
    converted_image = np.copy(gray_image)

    for box in boxes:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        converted_image[y1:y2, x1:x2][converted_image[y1:y2, x1:x2] == 1] = 0.5

    return converted_image

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    sam_model = 'vit_h'
    vitmatte_model = 'vit_b'
    tag2text_model = "ram"
    
    colors = [(255, 0, 0), (0, 255, 0)]
    markers = [1, 5]

    tag_stopwords = [
        "ground", "floor", "background", "blank", "backdrop",
        "spring", "summer", "autumn", "fall", "winter", 
        "snow", "rock", "rocks", "tree", "trees", 
        "pair", "color",
    ]

    print('Initializing models... Please wait...')

    predictor = init_segment_anything(sam_model)
    vitmatte = init_vitmatte(vitmatte_model)
    grounding_dino = dino_load_model(grounding_dino['config'], grounding_dino['weight'])
    rembg_session = new_session("isnet-general-use")
    ram_model = init_tag2text(tag2text_model)
    # processor, blip_model, lemma = init_blip(blip_model["weight"])

    def run_inference(input_x, erode_kernel_size, dilate_kernel_size, fg_box_threshold, fg_text_threshold, tr_box_threshold, tr_text_threshold, tr_caption = "glass, lens, crystal, diamond, bubble, bulb, web, grid"):
        
        # rembg
        image_pillow_raw = Image.fromarray(input_x)  # rgb
        rembg_out = remove(image_pillow_raw, session=rembg_session)
        rembg_out = rembg_out.convert("RGB")  # remove alpha channel

        # auto generate prompt
        # ram
        normalize = TS.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        transform = TS.Compose(
            [
                TS.Resize((384, 384)),
                TS.ToTensor(), 
                normalize
            ]
        )
        image_pillow = rembg_out.resize((384, 384))
        image_pillow = transform(image_pillow).unsqueeze(0).to(device)

        res = inference_ram.inference(image_pillow , ram_model)
        fg_caption = res[0].replace(" | ", ". ")
        tags_list = res[0].split(" | ")
        new_tags_list = []
        for tag in tags_list:
            if tag not in tag_stopwords:
                new_tags_list.append(tag)
        fg_caption = '. '.join(new_tags_list)
        print("TAG:", fg_caption)

        # # blip
        # blip_inputs = processor(rembg_out, return_tensors="pt").to(device, torch.float16)
        # blip_out = blip_model.generate(**blip_inputs)
        # caption = processor.decode(blip_out[0], skip_special_tokens=True)
        # print("CAPTION:", caption)

        # tags_list = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(caption)) if pos[0] == 'N']
        # tags_lemma = [lemma.lemmatize(w) for w in tags_list]
        # # fg_caption = '. '.join(map(str, tags_lemma))
        # # remove stopwords
        # tags_list = list(map(str, tags_lemma))
        # new_tags_list = []
        # for tag in tags_list:
        #     if tag not in tag_stopwords:
        #         new_tags_list.append(tag)
        # fg_caption = '. '.join(new_tags_list)
        # print("TAG:", fg_caption)

        # dino
        predictor.set_image(input_x)

        dino_transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_transformed, _ = dino_transform(Image.fromarray(input_x), None)
        
        point_coords, point_labels = None, None

        # fg_caption = "people."
        
        if fg_caption is not None and fg_caption != "": # This section has benefited from the contributions of neuromorph,thanks! 
            fg_boxes, logits, phrases = dino_predict(
                model=grounding_dino,
                image=image_transformed,
                caption=fg_caption,
                box_threshold=fg_box_threshold,
                text_threshold=fg_text_threshold,
                device=device)
            print(logits, phrases)
            if fg_boxes.shape[0] == 0:
                # no fg object detected
                transformed_boxes = None
            else:
                h, w, _ = input_x.shape
                fg_boxes = torch.Tensor(fg_boxes).to(device)
                fg_boxes = fg_boxes * torch.Tensor([w, h, w, h]).to(device)
                fg_boxes = box_convert(boxes=fg_boxes, in_fmt="cxcywh", out_fmt="xyxy")
                transformed_boxes = predictor.transform.apply_boxes_torch(fg_boxes, input_x.shape[:2])
        else:
            transformed_boxes = None
                    
        # predict segmentation according to the boxes
        masks, scores, logits = predictor.predict_torch(
            point_coords = point_coords,
            point_labels = point_labels,
            boxes = transformed_boxes,
            multimask_output = False,
        )
        masks = masks.cpu().detach().numpy()
        mask_all = np.ones((input_x.shape[0], input_x.shape[1], 3))
        for ann in masks:
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                mask_all[ann[0] == True, i] = color_mask[i]
        img = input_x / 255 * 0.3 + mask_all * 0.7
        
        # generate alpha matte
        torch.cuda.empty_cache()
        # mask = masks[0][0].astype(np.uint8)*255
        # use all mask
        masks_all = masks[0].copy()
        for ann in masks:
            masks_all[ann == True] = True
        mask = masks_all[0].astype(np.uint8)*255
        trimap = generate_trimap(mask, erode_kernel_size, dilate_kernel_size).astype(np.float32)
        trimap[trimap==128] = 0.5
        trimap[trimap==255] = 1
        
        boxes, logits, phrases = dino_predict(
            model=grounding_dino,
            image=image_transformed,
            caption= tr_caption,
            box_threshold=tr_box_threshold,
            text_threshold=tr_text_threshold,
            device=device)
        annotated_frame = dino_annotate(image_source=input_x, boxes=boxes, logits=logits, phrases=phrases)
        # 把annotated_frame的改成RGB
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        if boxes.shape[0] == 0:
            # no transparent object detected
            pass
        else:
            h, w, _ = input_x.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            trimap = convert_pixels(trimap, xyxy)

        input = {
            "image": torch.from_numpy(input_x).permute(2, 0, 1).unsqueeze(0)/255,
            "trimap": torch.from_numpy(trimap).unsqueeze(0).unsqueeze(0),
        }

        torch.cuda.empty_cache()
        alpha = vitmatte(input)['phas'].flatten(0,2)
        alpha = alpha.detach().cpu().numpy()
        
        # get a green background
        background = generate_checkerboard_image(input_x.shape[0], input_x.shape[1], 8)

        # calculate foreground with alpha blending
        foreground_alpha = input_x * np.expand_dims(alpha, axis=2).repeat(3,2)/255 + background * (1 - np.expand_dims(alpha, axis=2).repeat(3,2))/255

        # calculate foreground with mask
        foreground_mask = input_x * np.expand_dims(mask/255, axis=2).repeat(3,2)/255 + background * (1 - np.expand_dims(mask/255, axis=2).repeat(3,2))/255

        foreground_alpha[foreground_alpha>1] = 1
        foreground_mask[foreground_mask>1] = 1

        # return img, mask_all
        trimap[trimap==1] == 0.999

        # new background

        background_1 = cv2.imread('figs/sea.jpg')
        background_2 = cv2.imread('figs/forest.jpg')
        background_3 = cv2.imread('figs/sunny.jpg')

        background_1 = cv2.resize(background_1, (input_x.shape[1], input_x.shape[0]))
        background_2 = cv2.resize(background_2, (input_x.shape[1], input_x.shape[0]))
        background_3 = cv2.resize(background_3, (input_x.shape[1], input_x.shape[0]))

        # to RGB
        background_1 = cv2.cvtColor(background_1, cv2.COLOR_BGR2RGB)
        background_2 = cv2.cvtColor(background_2, cv2.COLOR_BGR2RGB)
        background_3 = cv2.cvtColor(background_3, cv2.COLOR_BGR2RGB)

        # use alpha blending
        new_bg_1 = input_x * np.expand_dims(alpha, axis=2).repeat(3,2)/255 + background_1 * (1 - np.expand_dims(alpha, axis=2).repeat(3,2))/255
        new_bg_2 = input_x * np.expand_dims(alpha, axis=2).repeat(3,2)/255 + background_2 * (1 - np.expand_dims(alpha, axis=2).repeat(3,2))/255
        new_bg_3 = input_x * np.expand_dims(alpha, axis=2).repeat(3,2)/255 + background_3 * (1 - np.expand_dims(alpha, axis=2).repeat(3,2))/255

        return  mask, alpha, foreground_mask, foreground_alpha, new_bg_1, new_bg_2, new_bg_3

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # <center>Matte Anything🐒 !
            """
        )
        with gr.Row().style(equal_height=True):
            with gr.Column():
                # input image
                original_image = gr.State(value="numpy")   # store original image without points, default None
                input_image = gr.Image(type="numpy", label="Input Image")              
                
                # run button
                button = gr.Button("Start!")

                # Trimap Settings
                with gr.Tab(label='Trimap Settings'):
                    gr.Markdown("Trimap Settings")
                    erode_kernel_size = gr.inputs.Slider(minimum=1, maximum=30, step=1, default=10, label="erode_kernel_size")
                    dilate_kernel_size = gr.inputs.Slider(minimum=1, maximum=30, step=1, default=10, label="dilate_kernel_size")
                
                # Input Text Settings
                with gr.Tab(label='Input Text Settings'):
                    gr.Markdown("Input Text Settings")
                    fg_box_threshold = gr.inputs.Slider(minimum=0.0, maximum=1.0, step=0.001, default=0.25, label="foreground_box_threshold")
                    fg_text_threshold = gr.inputs.Slider(minimum=0.0, maximum=1.0, step=0.001, default=0.25, label="foreground_text_threshold")

                # Transparency Settings
                with gr.Tab(label='Transparency Settings'):
                    gr.Markdown("Transparency Settings")
                    tr_caption = gr.inputs.Textbox(lines=1, default="glass.lens.crystal.diamond.bubble.bulb.web.grid", label="transparency input text")
                    tr_box_threshold = gr.inputs.Slider(minimum=0.0, maximum=1.0, step=0.005, default=0.5, label="transparency_box_threshold")
                    tr_text_threshold = gr.inputs.Slider(minimum=0.0, maximum=1.0, step=0.005, default=0.25, label="transparency_text_threshold")

            
            with gr.Column():

                # show the image with mask
                with gr.Tab(label='SAM Mask'):
                    mask = gr.Image(type='numpy')
                # with gr.Tab(label='Trimap'):
                #     trimap = gr.Image(type='numpy')
                with gr.Tab(label='Alpha Matte'):
                    alpha = gr.Image(type='numpy')
                # show only mask
                with gr.Tab(label='Foreground by SAM Mask'):
                    foreground_by_sam_mask = gr.Image(type='numpy')
                with gr.Tab(label='Refined by ViTMatte'):
                    refined_by_vitmatte = gr.Image(type='numpy')
                # with gr.Tab(label='Transparency Detection'):
                #     transparency = gr.Image(type='numpy')
                with gr.Tab(label='New Background 1'):
                    new_bg_1 = gr.Image(type='numpy')
                with gr.Tab(label='New Background 2'):
                    new_bg_2 = gr.Image(type='numpy')
                with gr.Tab(label='New Background 3'):
                    new_bg_3 = gr.Image(type='numpy')

        input_image.upload(
            store_img,
            [input_image],
            [original_image]
        )
        
        
        button.click(run_inference, inputs=[original_image, erode_kernel_size, dilate_kernel_size, fg_box_threshold, fg_text_threshold, tr_box_threshold, tr_text_threshold, \
                                             tr_caption],  outputs=[mask, alpha, foreground_by_sam_mask, refined_by_vitmatte, new_bg_1, new_bg_2, new_bg_3])
    
        with gr.Row():
            with gr.Column():
                background_image = gr.State(value=None)

    demo.launch()