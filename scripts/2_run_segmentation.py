import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def segment_images(image_folder, output_folder):
    """
    Runs semantic segmentation on all images in a folder and saves the results.
    """
    image_folder = Path(image_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load the model and processor
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")

    image_files = sorted([f for f in image_folder.glob('*.png')])

    for image_path in tqdm(image_files, desc="Segmenting images"):
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process to get semantic segmentation map
        result = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        
        # Save the result as a grayscale image
        output_image = Image.fromarray(result.cpu().numpy().astype('uint8'))
        
        output_filename = output_folder / f"{image_path.stem}_mask.png"
        output_image.save(output_filename)

if __name__ == '__main__':
    FRAME_INPUT_DIR = 'data/frames'
    MASK_OUTPUT_DIR = 'data/masks'

    segment_images(FRAME_INPUT_DIR, MASK_OUTPUT_DIR)