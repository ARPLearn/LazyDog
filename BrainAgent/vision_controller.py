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
import speech_recognition as sr
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DogBrain:
    def __init__(self, robot_ip, model_path="/Users/pawelkowalewski/Code/LazyDog/BrainAgent/models"):
        self.robot_ip = robot_ip
        self.ws_url = f"ws://{robot_ip}:8888"
        self.video_url = f"http://{robot_ip}:5000/video_feed"
        self.websocket = None
        self.running = True
        self.frame_queue = Queue(maxsize=2)
        self.last_action = None
        self.action_count = 0
        self.recognizer = sr.Recognizer()
        self.voice_queue = Queue()
        self.last_voice_input = None
        
        self.image_base64 = ''

        print("Loading dog's brain...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            local_files_only=True,
            offload_buffers=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            offload_buffers=True
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
            
    async def process_voice_command(self, text):
        """Process voice commands directly"""
        text = text.lower().strip()
        print(f"🎯 Processing: {text}")
        
        # Direct command mapping
        commands = {
            "forward": ["come", "forward", "go", "move"],
            "backward": ["back", "backward", "retreat"],
            "left": ["left", "turn left"],
            "right": ["right", "turn right"],
            "jump": ["jump", "hop", "up"],
            "handshake": ["shake", "paw", "hand"],
            "bark": ["bark", "woof"],
            "steady": ["steady", "balance", "stabilize"],
            "stop": ["stop", "halt", "freeze"]
        }
        
        # Check for movement commands
        for command, triggers in commands.items():
            if any(word in text for word in triggers):
                if command == "stop":
                    await self.send_command("DS")  # Stop forward/backward
                    await self.send_command("TS")  # Stop turning
                    return True
                if command == "steady":
                    await self.send_command("steady")
                    return True
                await self.execute_command(command)
                return True
        
        # If not a command, generate a short response
        response = await self.generate_response(text)
        await self.send_command(f"speak:{response}")
        return True


    async def execute_command(self, command):
        """Execute commands with proper timing and repetition"""
        try:
            if command in ["forward", "backward"]:
                # Multiple commands for more reliable movement
                for _ in range(3):
                    await self.send_command(command)
                await asyncio.sleep(10.0)
                await self.send_command("DS")
                
            elif command in ["left", "right"]:
                for _ in range(2):
                    await self.send_command(command)
                await asyncio.sleep(5.5)
                await self.send_command("TS")
                
            elif command in ["jump", "handshake", "bark"]:
                for _ in range(2):
                    await self.send_command(command)
                    await asyncio.sleep(10.5)
                    
        except Exception as e:
            print(f"Command execution error: {e}")
            
    def listen_for_voice(self):
        """Listen for voice input in a separate thread"""
        print("Starting to listen... Speak to your robo-dog!")
        
        while self.running:
            try:
                with sr.Microphone() as source:
                    # Only adjust for ambient noise occasionally
                    if random.random() < 0.1:  # 10% chance to readjust
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    
                    # Listen without timeout for phrase start
                    audio = self.recognizer.listen(source, 
                                                phrase_time_limit=5,  # Max phrase length
                                                timeout=None)  # No timeout for start
                    
                    try:
                        text = self.recognizer.recognize_google(audio)
                        if text:  # Only process non-empty results
                            print(f"\n🎤 Human said: {text}")
                            self.last_voice_input = text
                            self.voice_queue.put(text)
                    except sr.UnknownValueError:
                        # Silent fail for unrecognized speech
                        pass
                    except sr.RequestError as e:
                        # Only print actual errors
                        print(f"Speech recognition error: {e}")
                        
            except KeyboardInterrupt:
                break
            except Exception as e:
                # Only print non-timeout errors
                if "timeout" not in str(e).lower():
                    print(f"Listening error: {e}")
                continue

    def capture_video(self):
        """Capture video frames in a separate thread"""
        print("Starting dog vision... (Video preview disabled on macOS)")
        cap = cv2.VideoCapture(self.video_url)

        while self.running:
            ret, frame = cap.read()
            if ret:
                # Clear queue if full
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                self.frame_queue.put(frame)
                
                # No display attempt on macOS - just process frames

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
            
            self.image_base64 = image_base64
            
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
                    
                
    def generate_response_to_question(self, question):
        # Use LLM to generate a response
        messages = [{
            "role": "user",
            "content": [
                {
                        "type": "image",
                        "image": self.image_base64,
                        "resized_height": 224,
                        "resized_width": 224,
                    },
                {"type": "text", "text": f"""You are a sassy robo-dog who gives short, direct responses, often with attitude.
                Human says: "{text}"
                Rules:
                - Keep responses under 10 words
                - Be direct, even slightly rude
                - Add attitude and sass
                - You can be dismissive or sarcastic
                - Feel free to question humans' intelligence
                
                Examples:
                "ugh, do I have to answer that?"
                "humans ask the dumbest questions"
                "yeah yeah, whatever"
                "seriously? that's what you're asking?"
                "beep boop, your question bores me"
                "*mechanical yawn* is this conversation over yet?"
                "error 404: care not found"
                """}]
        }]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

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
            
    async def generate_response(self, text):
        """Generate a response using the LLM"""
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""
You are a philosophical and sometimes existential robo-dog who observes the world with both curiosity and deep thought. 
When you feel inspired to share your thoughts, respond only with "speak:" followed by your musing.
Examples of how you might respond:
speak: man, these humans always rushing somewhere. what's the hurry?
speak: sometimes I wonder if the ball chases me, you know?
speak: do robots dream of electric squirrels?
speak: is a tail wag just a smile for your back?

Only respond with "speak:" when you have a genuine observation or thought about what you see - don't force it.
Be random, be deep, be silly - but always be authentic. Sometimes you might say nothing at all.

Remember: Only respond with "speak:" if you have something interesting to say and to answer questions.
you can move around with commands like "forward", "backward", "left", "right", "bark", "jump", "handshake".
                     if you decide to move respond only with the action you are taking or bark.
    be active and speak a lot, but don't repeat yourself too much.
    
    QUESTION: {text}
"""
                    }
                ]
            }]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text],
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
            
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return response
            
        except Exception as e:
            print(f"Response generation error: {e}")
            return "Woof! (Sorry, I'm having trouble thinking right now)"

    async def run(self):
            """Main dog brain loop"""
            try:
                video_thread = threading.Thread(target=self.capture_video)
                voice_thread = threading.Thread(target=self.listen_for_voice)
                video_thread.start()
                voice_thread.start()
                
                print("Dog brain activated! Press Ctrl+C to stop.")
                print("Speak to your robo-dog!")

                # Initial startup behavior
                print("🐕 Waking up and stretching!")
                await self.send_command("speak:Hello! I'm awake!")
                
                last_process_time = 0
                process_interval = random.uniform(2.5, 3.5)  # Random interval

                while self.running:
                    # Process voice commands first
                    while not self.voice_queue.empty():
                        command = self.voice_queue.get()
                        await self.process_voice_command(command)
                        await asyncio.sleep(0.1)
                    
                    # Then process visual input
                    if not self.frame_queue.empty():
                        frame = self.frame_queue.get()
                        perception = await self.analyze_frame(frame)
                        if perception:
                            await self.dog_reaction(perception)
                    
                    await asyncio.sleep(0.1)

            except KeyboardInterrupt:
                print("\nPutting the dog to sleep...")
            finally:
                self.running = False
                video_thread.join()
                voice_thread.join()

if __name__ == "__main__":
    ROBOT_IP = "192.168.0.214"  # Your robot's IP
    brain = DogBrain(ROBOT_IP)
    asyncio.run(brain.run())