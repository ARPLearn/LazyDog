from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

def test_llm():
    print("Testing with minimal settings...")
    print("Device available:", "cuda" if torch.cuda.is_available() else "cpu")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "D:/Code/WavegoAgent/models",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        local_files_only=True,
    )
    
    processor = AutoProcessor.from_pretrained(
        "D:/Code/WavegoAgent/models",
        local_files_only=True
    )

    # Use small image size
    processor.image_processor.min_pixels = 224 * 224
    processor.image_processor.max_pixels = 224 * 224

    # Test message
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    "resized_height": 224,
                    "resized_width": 224,
                },
                {"type": "text", "text": "What do you see in this image? Answer in one short sentence."}
            ]
        }
    ]

    print("Processing input...")
    # Process both text and image
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    print("Starting generation...")
    
    # Generate with simple settings
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=20,    # Short response
        num_beams=1,          # No beam search
        do_sample=True,       # Enable sampling
        temperature=0.7,      # Slightly random
        top_p=0.9,           # Nucleus sampling
    )
    
    # Trim the input prompt from the output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    print("\nOutput:", output_text[0])

if __name__ == "__main__":
    test_llm()