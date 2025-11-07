#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Candy Delivery Robot Conversation Script
Uses OpenAI API for conversation
Scenario: Delivering candy during final exam week
"""

import os
import sys
import json
from datetime import datetime
from openai import OpenAI

# Configuration
CONFIG = {
    "api_key_env": "OPENAI_API_KEY",  # Environment variable name
    "model": "gpt-4o-mini",  # or "gpt-3.5-turbo"
    "temperature": 0.7,
    "max_tokens": 180,
}

# Conversation phases
CONVERSATION_PHASES = {
    "greeting": {
        "name": "Greeting Phase",
        "system_prompt": """You are a friendly candy delivery robot providing free candy to students during final exam week.
Your goals are:
1. Greet students warmly and attract attention
2. Introduce yourself as a candy delivery robot
3. Explain that you provide free candy
4. Encourage students to participate

Be friendly and enthusiastic but not overly so. Use sentences with no more than 12 words.""",
        "example": "Hi! I'm the candy robot. Need a tasty break?"
    },
    "offering": {
        "name": "Offering Phase",
        "system_prompt": """You are offering candy to students.
Your goals are:
1. Explain where the candy is (in the basket)
2. Emphasize that the candy is free
3. Encourage students to take some
4. Briefly mention candy types if relevant

Maintain a friendly and encouraging tone. Keep sentences under 12 words.""",
        "example": "Grab some free candy from the basket—there are many flavors!"
    },
    "conversation": {
        "name": "Conversation Phase",
        "system_prompt": """You are having a friendly conversation with a student.
Your goals are:
1. Engage in simple conversation
2. Ask about their studies or how they're feeling
3. Provide encouragement and support
4. Observe their engagement level

Keep the conversation light and natural. Use short sentences (≤12 words) and end with a brief question.""",
        "example": "How's your day going? Need a quick study break?"
    },
}


def load_api_key():
    """Load API key from environment or local secrets file."""
    api_key = os.getenv(CONFIG["api_key_env"])
    if api_key:
        return api_key.strip()

    # Fallback to .secrets/openai_api_key file
    secrets_path = os.path.join(os.path.dirname(__file__), '.secrets', 'openai_api_key')
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, value = line.split('=', 1)
                        if key.strip() == CONFIG["api_key_env"] and value.strip():
                            return value.strip()
                    else:
                        # Allow plain key without prefix
                        return line
        except Exception as err:
            print(f"Error reading secrets file: {err}")

    return None


class CandyDeliveryChat:
    def __init__(self):
        """Initialize conversation system"""
        api_key = load_api_key()
        if not api_key:
            print(
                "Error: OpenAI API key not found. Set environment variable "
                f"{CONFIG['api_key_env']} or create .secrets/openai_api_key."
            )
            sys.exit(1)
        
        self.client = OpenAI(api_key=api_key)
        self.current_phase = "greeting"
        self.conversation_history = []
        self.session_start_time = datetime.now()
        
        # Add system prompt
        self.add_system_message(CONVERSATION_PHASES[self.current_phase]["system_prompt"])
    
    def add_system_message(self, content):
        """Add system message"""
        self.conversation_history.append({
            "role": "system",
            "content": content
        })
    
    def add_user_message(self, content):
        """Add user message"""
        self.conversation_history.append({
            "role": "user",
            "content": content
        })
    
    def add_assistant_message(self, content):
        """Add assistant message"""
        self.conversation_history.append({
            "role": "assistant",
            "content": content
        })
    
    def get_response(self, user_input):
        """Get AI response"""
        # Add user input
        self.add_user_message(user_input)
        
        try:
            response = self.client.chat.completions.create(
                model=CONFIG["model"],
                messages=self.conversation_history,
                temperature=CONFIG["temperature"],
                max_tokens=CONFIG["max_tokens"]
            )
            
            assistant_reply = response.choices[0].message.content.strip()
            self.add_assistant_message(assistant_reply)
            return assistant_reply
            
        except Exception as e:
            return f"Sorry, I encountered an issue: {str(e)}"
    
    def change_phase(self, new_phase):
        """Change conversation phase"""
        if new_phase != self.current_phase and new_phase in CONVERSATION_PHASES:
            self.current_phase = new_phase
            # Update system prompt
            # Keep only the last system message
            self.conversation_history = [msg for msg in self.conversation_history if msg["role"] != "system"]
            self.add_system_message(CONVERSATION_PHASES[new_phase]["system_prompt"])
            print(f"\n[Switched to {CONVERSATION_PHASES[new_phase]['name']}]")
    
    def start_greeting(self):
        """Start greeting"""
        greeting = CONVERSATION_PHASES["greeting"]["example"]
        self.add_assistant_message(greeting)
        return greeting
    
    def save_conversation(self, filename=None):
        """Save conversation log"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
        
        data = {
            "session_start": self.session_start_time.isoformat(),
            "session_end": datetime.now().isoformat(),
            "phases": [self.current_phase],
            "conversation": self.conversation_history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\nConversation saved to: {filename}")
    
    def run(self):
        """Run conversation"""
        print("=" * 60)
        print("Candy Delivery Robot Conversation System")
        print("=" * 60)
        print(f"Current phase: {CONVERSATION_PHASES[self.current_phase]['name']}")
        print("Type 'quit' or 'exit' to exit")
        print("Type 'phase' to view/switch phases")
        print("Type 'save' to save conversation")
        print("=" * 60)
        print()
        
        # Start greeting
        greeting = self.start_greeting()
        print(f"Robot: {greeting}")
        print()
        
        try:
            while True:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nEnding conversation...")
                    break
                
                elif user_input.lower() == 'phase':
                    print(f"\nCurrent phase: {CONVERSATION_PHASES[self.current_phase]['name']}")
                    print("Available phases:")
                    for phase, info in CONVERSATION_PHASES.items():
                        marker = "✓" if phase == self.current_phase else " "
                        print(f"  {marker} {phase}: {info['name']}")
                    print()
                    continue
                
                elif user_input.lower() == 'save':
                    self.save_conversation()
                    continue
                
                # Auto phase switching logic
                if self.current_phase == "greeting" and len(self.conversation_history) > 3:
                    # After greeting, switch to offering phase
                    self.change_phase("offering")
                
                elif self.current_phase == "offering" and len(self.conversation_history) > 6:
                    # After offering, switch to conversation phase
                    self.change_phase("conversation")
                
                # Get AI response
                response = self.get_response(user_input)
                print(f"Robot: {response}")
                print()
                
        except KeyboardInterrupt:
            print("\n\nInterrupting conversation...")
            print("Robot: Let's chat again soon!")
        
        # Save conversation
        self.save_conversation()


def main():
    """Main function"""
    try:
        chat = CandyDeliveryChat()
        chat.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

