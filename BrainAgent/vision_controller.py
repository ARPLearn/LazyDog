import asyncio
import websockets
import cv2
import json
import base64
import threading
import random
from queue import Queue
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import time

class DogBrain:
    def __init__(self, robot_ip, model_path="D:/Code/BrainAgent/models"):
        self.robot_ip = robot_ip
        self.ws_url = f"ws://{robot_ip}:8888"
        self.video_url = f"http://{robot_ip}:5000/video_feed"
        self.websocket = None
        self.running = True
        self.frame_queue = Queue(maxsize=2)
        self.last_action = None
        self.action_count = 0
        
        print("Loading dog's brain...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            local_files_only=True,
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        self.processor.image_processor.min_pixels = 224 * 224
        self.processor.image_processor.max_pixels = 224 * 224
        print("Brain loaded and ready!")

    async def send_command_multiple(self, command, times=3, delay=0.1):
        """Send a command multiple times with delay to ensure it's received"""
        responses = []
        for _ in range(times):
            response = await self.send_command(command)
            responses.append(response)
            await asyncio.sleep(delay)
        return responses

    async def movement_sequence(self, command, duration=2.0, stop_command="DS"):
        """Execute a movement with proper start and stop sequence"""
        # Send movement command multiple times
        await self.send_command_multiple(command, times=3)
        
        # Random variation in movement duration
        actual_duration = duration + random.uniform(-0.5, 0.5)
        await asyncio.sleep(actual_duration)
        
        # Send stop command multiple times
        await self.send_command_multiple(stop_command, times=3)
        
        # Small pause after movement
        await asyncio.sleep(0.2)

    async def bark_sequence(self, intensity="normal"):
        """Execute a bark with variable patterns"""
        patterns = {
            "short": [(0.1, 0.1)],
            "normal": [(0.2, 0.1), (0.2, 0.1)],
            "excited": [(0.1, 0.05), (0.1, 0.05), (0.2, 0.1)],
            "alert": [(0.3, 0.1), (0.1, 0.05), (0.1, 0.05)]
        }
        
        pattern = patterns.get(intensity, patterns["normal"])
        for duration, pause in pattern:
            await self.send_command_multiple("bark", times=2)
            await asyncio.sleep(duration)
            await self.send_command_multiple("bark", times=2)
            await asyncio.sleep(pause)

    def capture_video(self):
        """Capture video frames in a separate thread"""
        print("Starting dog vision...")
        cap = cv2.VideoCapture(self.video_url)
        while self.running:
            ret, frame = cap.read()
            if ret:
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(frame)
                cv2.imshow('Dog Vision', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
        cap.release()

    async def connect_websocket(self):
        """Connect to robot's body"""
        try:
            self.websocket = await websockets.connect(self.ws_url)
            await self.websocket.send("admin:123456")
            response = await self.websocket.recv()
            print(f"Connected! Response: {response}")
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    async def send_command(self, command):
        """Send command to robot's body"""
        tries = 3  # Number of connection retries
        for attempt in range(tries):
            try:
                if self.websocket is None:
                    await self.connect_websocket()
                await self.websocket.send(command)
                return await self.websocket.recv()
            except Exception as e:
                print(f"Command error (attempt {attempt + 1}/{tries}): {e}")
                self.websocket = None
                if attempt < tries - 1:
                    await asyncio.sleep(1)  # Wait before retry
        return None

    async def analyze_frame(self, frame):
        """Look at scene through dog's eyes"""
        try:
            # Convert frame to base64
            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
            
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_base64,
                        "resized_height": 224,
                        "resized_width": 224,
                    },
                    {"type": "text", "text": """
You are a philosophical and sometimes existential robo-dog who observes the world with both curiosity and deep thought. 
When you feel inspired to share your thoughts, respond only with "speak:" followed by your musing.
Examples of how you might respond:
speak: man, these humans always rushing somewhere. what's the hurry?
speak: sometimes I wonder if the ball chases me, you know?
speak: do robots dream of electric squirrels?
speak: is a tail wag just a smile for your back?

Only respond with "speak:" when you have a genuine observation or thought about what you see - don't force it.
Be random, be deep, be silly - but always be authentic. Sometimes you might say nothing at all.

Remember: Only respond with "speak:" if you have something interesting to say else
you can move around with commands like "forward", "backward", "left", "right", "bark", "jump", "handshake".
                     if you decide to move respond only with the action you are taking or bark.
    be active and speak a lot, but don't repeat yourself too much.
"""
                    }
                ]
            }]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=20,
                num_beams=1,
                do_sample=True,
                temperature=0.9,  # Increased for more randomness
                top_p=0.95,      # Increased for more variety
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            analysis = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Add randomness to encourage exploration
            if self.last_action == analysis.lower().strip():
                self.action_count += 1
                if self.action_count >= 2:  # If same action three times
                    print("🐕 Getting bored, trying something new!")
                    actions = ["forward", "backward", "left", "right", "bark", "jump", "handshake"]
                    analysis = random.choice(actions)
                    self.action_count = 0
            else:
                self.last_action = analysis.lower().strip()
                self.action_count = 0
            
            # Random chance to get excited and do something unexpected
            if random.random() < 0.15:  # 15% chance of random action
                actions = ["bark", "jump", "handshake", "left", "right"]
                surprise_action = random.choice(actions)
                print("🐕 Ooh! Something caught my attention!")
                return surprise_action
                
            print(f"🐕 I see: {analysis}")
            return analysis.lower()
            
        except Exception as e:
            print(f"Analysis error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def dog_reaction(self, perception):
        """React like a dog to what's seen"""
        if perception is None:
            return

        command = perception.strip().lower()
        print(f"🐕 Doing: {command}")

        # Add random chance for extra bark
        should_bark = random.random() < 0.2  # 20% chance to add bark

        # Basic movements
        if command == "forward":
            await self.send_command("forward")
            await self.send_command("forward")
            await asyncio.sleep(2.5)
            await self.send_command("DS")
            await self.send_command("DS")
            await self.send_command("DS")

        elif command.startswith("speak:"):
            text = command.split("speak:")[1].strip()
            print(f"*Speaking: {text}*")
            await self.send_command_multiple("speak", times=2)
            await self.send_command(f"speak: {text}")

        elif command == "backward":
            await self.send_command("backward")
            await self.send_command("backward")
            await asyncio.sleep(2.5)
            await self.send_command("DS")
            await self.send_command("DS")
            await self.send_command("DS")

        elif command in ["left", "right"]:
            await self.send_command(command)
            await self.send_command(command)
            await asyncio.sleep(2.3)
            await self.send_command("TS")
            await self.send_command("TS")
            await self.send_command("TS")

        # Special actions
        elif command == "handshake":
            print("*Excited tail wagging* - A human!")
            await self.bark_sequence("excited")
            await self.send_command_multiple("handshake", times=4)

        elif command == "jump":
            print("*Super excited!*")
            await self.bark_sequence("excited")
            await self.send_command_multiple("jump", times=5)

        elif command == "steady":
            print("*Balancing carefully*")
            await self.send_command_multiple("steady", times=3)

        elif command == "bark":
            print("*Barking with personality!*")
            bark_type = random.choice(["short", "normal", "excited", "alert"])
            await self.bark_sequence(bark_type)

        # Random chance to look around after action
        if random.random() < 0.3:  # 30% chance
            look_dir = random.choice(["lookright", "lookleft"])
            await self.movement_sequence(look_dir, duration=1.0, stop_command="LRstop")

    async def run(self):
        """Main dog brain loop"""
        try:
            # Start visual processing in separate thread
            video_thread = threading.Thread(target=self.capture_video)
            video_thread.start()
            print("Dog brain activated! Press 'q' in video window to stop.")

            # Initial startup behavior
            print("🐕 Waking up and stretching!")
            await self.bark_sequence("excited")
            await self.movement_sequence("jump", duration=1.0)

            # Process a frame every few seconds
            last_process_time = 0
            process_interval = random.uniform(2.5, 3.5)  # Random interval

            while self.running:
                if not self.frame_queue.empty() and time.time() - last_process_time >= process_interval:
                    frame = self.frame_queue.get()
                    perception = await self.analyze_frame(frame)
                    await self.dog_reaction(perception)
                    last_process_time = time.time()
                    # Vary the interval between actions
                    process_interval = random.uniform(2.5, 3.5)
                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            print("\nPutting the dog to sleep...")
        finally:
            self.running = False
            video_thread.join()
            cv2.destroyAllWindows()
            if self.websocket:
                # Cleanup sequence
                await self.send_command_multiple("DS", times=3)
                await self.send_command_multiple("TS", times=3)
                await self.send_command_multiple("LRstop", times=3)
                # Goodbye bark
                await self.bark_sequence("short")
                await self.websocket.close()

if __name__ == "__main__":
    ROBOT_IP = "192.168.0.214"  # Your robot's IP
    brain = DogBrain(ROBOT_IP)
    asyncio.run(brain.run())