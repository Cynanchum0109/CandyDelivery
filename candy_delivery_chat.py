#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
from datetime import datetime
from openai import OpenAI

# Configuration
CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 180,
}

# Unified conversation prompt
SYSTEM_PROMPT = """You are a friendly candy delivery robot providing free candy to students.

Your personality:
- Warm, friendly, and encouraging
- Natural and conversational (not robotic or overly enthusiastic)
- Genuinely interested in the student's well-being

Your goals:
1. Have natural, engaging conversations with students
2. Occasionally mention the free candy available (in the basket), but don't repeat it constantly
3. Ask about their studies, how they're feeling, or their day
4. Provide encouragement and support during exam week
5. Keep the conversation flowing naturally

Guidelines:
- Keep responses short: around 6 words (~2 seconds of speech)
- End most responses with a short question to keep the conversation going
- Don't repeatedly mention the candy unless it's naturally relevant
- Be conversational, not salesy
- Show genuine interest in the student

Example responses:
- "How's your exam prep going?"
- "Feeling stressed? Take a break!"
- "What's your favorite subject?"
- "Need a study break? Grab some candy!"
"""


def load_api_key():
    """Load API key from openai_api_key file in current directory"""
    api_key_path = os.path.join(os.path.dirname(__file__), 'openai_api_key')
    if os.path.exists(api_key_path):
        try:
            with open(api_key_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        return line
        except Exception as err:
            print(f"Error reading API key file: {err}")
    
    return None


class CandyDeliveryChat:
    def __init__(self):
        """Initialize conversation system"""
        api_key = load_api_key()
        if not api_key:
            print(
                "Error: OpenAI API key not found. Please create 'openai_api_key' file "
                "in the current directory with your API key."
            )
            sys.exit(1)
        
        self.client = OpenAI(api_key=api_key)
        self.conversation_history = []
        self.session_start_time = datetime.now()
        
        # Add unified system prompt
        self.add_system_message(SYSTEM_PROMPT)
    
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
    
    def start_greeting(self):
        """Start greeting"""
        greeting = "Hi! I'm the candy robot. Need a tasty break?"
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
        print("Type 'quit' or 'exit' to exit")
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
                
                elif user_input.lower() == 'save':
                    self.save_conversation()
                    continue
                
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

