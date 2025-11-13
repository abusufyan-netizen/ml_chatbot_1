import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import nltk
import pandas as pd
import speech_recognition as sr
import pyttsx3
import threading
import json
import os
import random
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize NLTK - make sure punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class AdvancedChatbot:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Chatbot - Local Dataset")
        self.root.geometry("1400x800")
        self.root.configure(bg='#2b2b2b')
        
        # Initialize speech engine
        self.speech_engine = pyttsx3.init()
        self.speech_engine.setProperty('rate', 150)
        
        # Load dataset
        self.dataset = self.load_dataset()
        self.vectorizer = TfidfVectorizer()
        self.setup_similarity_model()
        
        # Chat history
        self.chat_history = []
        self.current_chat = []
        
        self.setup_ui()
        self.load_chat_history()
    
    def load_dataset(self):
        """Load the CSV dataset"""
        try:
            df = pd.read_csv('dataset.csv')
            print(f"Dataset loaded with {len(df)} records")
            return df
        except FileNotFoundError:
            print("Dataset file not found. Creating sample dataset...")
            return self.create_sample_dataset()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return self.create_sample_dataset()
    
    def create_sample_dataset(self):
        """Create a sample dataset if file doesn't exist"""
        data = {
            'question': [
                'hello', 'hi', 'how are you', 'what is your name', 'what can you do',
                'tell me a joke', 'what is python', 'how to learn coding',
                'what is machine learning', 'what is artificial intelligence',
                'good morning', 'good afternoon', 'good evening', 'who created you',
                'what time is it', 'how to make coffee', 'what is photosynthesis',
                'benefits of exercise', 'book recommendations', 'how to play guitar'
            ],
            'answer': [
                'Hello! How can I assist you today?',
                'Hi there! What can I help you with?',
                "I'm doing great, thank you! How can I assist you?",
                "I'm an AI chatbot designed to help with your questions.",
                'I can answer questions, provide information, help with calculations, and more!',
                "Why don't scientists trust atoms? Because they make up everything!",
                'Python is a high-level programming language known for its simplicity and readability.',
                'Start with basic programming concepts, practice regularly, and build small projects.',
                'Machine learning is a subset of AI that enables computers to learn from data.',
                'AI is the simulation of human intelligence in machines.',
                'Good morning! How are you doing today?',
                'Good afternoon! How can I help you?',
                'Good evening! What would you like to know?',
                'I was created by a developer to assist with various tasks.',
                'I cannot access real-time information, but you can check your device clock.',
                'Boil water, add coffee grounds, pour hot water, and let it brew.',
                'Photosynthesis is how plants convert sunlight into energy.',
                'Exercise improves health, boosts mood, and increases energy levels.',
                'I recommend "Atomic Habits", "The Alchemist", and "Sapiens".',
                'Start with basic chords, practice regularly, and learn simple songs.'
            ],
            'category': [
                'greeting', 'greeting', 'greeting', 'about', 'about',
                'fun', 'programming', 'advice', 'technology', 'technology',
                'greeting', 'greeting', 'greeting', 'about', 'information',
                'cooking', 'science', 'health', 'entertainment', 'hobbies'
            ]
        }
        return pd.DataFrame(data)
    
    def setup_similarity_model(self):
        """Setup TF-IDF vectorizer for similarity matching"""
        try:
            questions = self.dataset['question'].tolist()
            self.tfidf_matrix = self.vectorizer.fit_transform(questions)
        except Exception as e:
            print(f"Error setting up similarity model: {e}")
    
    def find_best_match(self, user_input):
        """Find the best matching question in the dataset"""
        try:
            # Transform user input
            user_tfidf = self.vectorizer.transform([user_input])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(user_tfidf, self.tfidf_matrix)
            best_match_idx = similarities.argmax()
            best_similarity = similarities[0, best_match_idx]
            
            # Return best match if similarity is above threshold
            if best_similarity > 0.3:
                return self.dataset.iloc[best_match_idx]['answer']
            else:
                return None
                
        except Exception as e:
            print(f"Error in similarity matching: {e}")
            return None
    
    def setup_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left sidebar (1/4 width)
        left_frame = tk.Frame(main_frame, bg='#3c3c3c', width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)
        
        # Right main content (3/4 width)
        right_frame = tk.Frame(main_frame, bg='#2b2b2b')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_left_sidebar(left_frame)
        self.setup_right_content(right_frame)
    
    def setup_left_sidebar(self, parent):
        # Title
        title_label = tk.Label(parent, text="Chat History", font=('Arial', 16, 'bold'), 
                              bg='#3c3c3c', fg='white')
        title_label.pack(pady=20)
        
        # New Chat button
        new_chat_btn = tk.Button(parent, text="+ New Chat", font=('Arial', 12),
                                bg='#4CAF50', fg='white', relief='flat',
                                command=self.start_new_chat)
        new_chat_btn.pack(pady=10, padx=20, fill=tk.X)
        
        # Dataset Status
        status_text = f"Dataset: ✅ {len(self.dataset)} records"
        status_label = tk.Label(parent, text=status_text, font=('Arial', 10),
                               bg='#3c3c3c', fg='#4CAF50')
        status_label.pack(pady=5)
        
        # History list
        history_frame = tk.Frame(parent, bg='#3c3c3c')
        history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.history_listbox = tk.Listbox(history_frame, bg='#2b2b2b', fg='white',
                                         selectbackground='#4CAF50', font=('Arial', 10))
        self.history_listbox.pack(fill=tk.BOTH, expand=True)
        self.history_listbox.bind('<<ListboxSelect>>', self.load_selected_chat)
        
        # Dataset Management button
        dataset_btn = tk.Button(parent, text="Manage Dataset", font=('Arial', 10),
                               bg='#FF9800', fg='white', relief='flat',
                               command=self.show_dataset_management)
        dataset_btn.pack(pady=10, padx=20, fill=tk.X)
    
    def setup_right_content(self, parent):
        # Chat display area
        chat_frame = tk.Frame(parent, bg='#2b2b2b')
        chat_frame.pack(fill=tk.BOTH, expand=True)
        
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD,
                                                     bg='#1e1e1e', fg='white',
                                                     font=('Arial', 11),
                                                     state=tk.DISABLED)
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Input area
        input_frame = tk.Frame(parent, bg='#2b2b2b')
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Suggestions label
        self.suggestions_label = tk.Label(input_frame, text="Suggestions: ", 
                                         bg='#2b2b2b', fg='#888888', font=('Arial', 9))
        self.suggestions_label.pack(anchor=tk.W)
        
        # Input frame with button
        input_button_frame = tk.Frame(input_frame, bg='#2b2b2b')
        input_button_frame.pack(fill=tk.X)
        
        self.input_entry = tk.Entry(input_button_frame, font=('Arial', 12),
                                   bg='#3c3c3c', fg='white', insertbackground='white')
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_entry.bind('<KeyRelease>', self.on_input_change)
        self.input_entry.bind('<Return>', self.send_message)
        
        # Buttons frame
        button_frame = tk.Frame(input_button_frame, bg='#2b2b2b')
        button_frame.pack(side=tk.RIGHT)
        
        send_btn = tk.Button(button_frame, text="Send", font=('Arial', 10),
                            bg='#4CAF50', fg='white', command=self.send_message)
        send_btn.pack(side=tk.LEFT, padx=5)
        
        voice_btn = tk.Button(button_frame, text="🎤", font=('Arial', 12),
                             bg='#2196F3', fg='white', command=self.start_voice_input)
        voice_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(button_frame, text="Clear", font=('Arial', 10),
                             bg='#f44336', fg='white', command=self.clear_chat)
        clear_btn.pack(side=tk.LEFT, padx=5)
    
    def on_input_change(self, event=None):
        current_text = self.input_entry.get()
        if current_text:
            suggestions = self.get_suggestions(current_text)
            self.suggestions_label.config(text=f"Suggestions: {', '.join(suggestions)}")
        else:
            self.suggestions_label.config(text="Suggestions: ")
    
    def get_suggestions(self, query):
        """Get suggestions based on input"""
        query_lower = query.lower()
        
        # Get categories from dataset
        categories = self.dataset['category'].unique().tolist()
        
        # Filter dataset for similar questions
        similar_questions = []
        for question in self.dataset['question']:
            if query_lower in question.lower():
                similar_questions.append(question)
        
        return similar_questions[:3] if similar_questions else ["Ask me anything!", "Try a question", "Need help?"]
    
    def send_message(self, event=None):
        user_input = self.input_entry.get().strip()
        if not user_input:
            return
        
        self.input_entry.delete(0, tk.END)
        self.display_message(f"You: {user_input}", "user")
        
        # Process message in thread
        threading.Thread(target=self.process_message, args=(user_input,), daemon=True).start()
    
    def process_message(self, user_input):
        response = self.generate_response(user_input)
        self.display_message(f"Bot: {response}", "bot")
        self.speak_response(response)
    
    def generate_response(self, user_input):
        """Generate response using the dataset"""
        # Try to find best match in dataset
        best_match = self.find_best_match(user_input)
        
        if best_match:
            return best_match
        
        # Fallback responses for common queries
        input_lower = user_input.lower()
        
        # Greeting detection
        if any(word in input_lower for word in ['hello', 'hi', 'hey', 'hola']):
            return random.choice([
                "Hello! How can I help you today?",
                "Hi there! What would you like to know?",
                "Hey! How can I assist you?"
            ])
        
        # Time related
        elif any(word in input_lower for word in ['time', 'clock', 'hour']):
            return f"The current time is {datetime.now().strftime('%H:%M:%S')}"
        
        # Date related
        elif any(word in input_lower for word in ['date', 'day', 'today']):
            return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}"
        
        # Thank you
        elif any(word in input_lower for word in ['thank', 'thanks']):
            return "You're welcome! Is there anything else I can help with?"
        
        # Goodbye
        elif any(word in input_lower for word in ['bye', 'goodbye', 'exit', 'quit']):
            return "Goodbye! Feel free to chat again anytime!"
        
        # Default response
        else:
            return random.choice([
                "That's an interesting question! I'm still learning, but I'll try to help.",
                "I understand you're asking about that. Could you try rephrasing your question?",
                "I'm not sure I have the answer to that in my dataset yet.",
                "That's a great question! My knowledge is based on my training data.",
                "I'm constantly learning. Could you ask me something else?"
            ])
    
    def start_voice_input(self):
        """Start voice input in English"""
        threading.Thread(target=self.process_voice_input, daemon=True).start()
    
    def process_voice_input(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            try:
                self.display_message("Bot: Listening... Speak now", "bot")
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio, language='en-US')
                self.input_entry.delete(0, tk.END)
                self.input_entry.insert(0, text)
                self.display_message(f"You (Voice): {text}", "user")
                self.process_message(text)
            except sr.UnknownValueError:
                self.display_message("Bot: Sorry, I couldn't understand the audio", "bot")
            except sr.RequestError:
                self.display_message("Bot: Speech recognition service error", "bot")
            except sr.WaitTimeoutError:
                self.display_message("Bot: No speech detected", "bot")
    
    def speak_response(self, text):
        """Convert text to speech in English"""
        try:
            self.speech_engine.say(text)
            self.speech_engine.runAndWait()
        except:
            pass
    
    def display_message(self, message, sender):
        self.root.after(0, self._update_display, message, sender)
    
    def _update_display(self, message, sender):
        self.chat_display.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Configure tags for different senders
        self.chat_display.tag_config('user', foreground='#4CAF50')
        self.chat_display.tag_config('bot', foreground='#2196F3')
        self.chat_display.tag_config('timestamp', foreground='#888888')
        
        # Insert message with formatting
        self.chat_display.insert(tk.END, f"[{timestamp}] ", 'timestamp')
        self.chat_display.insert(tk.END, f"{message}\n\n", sender)
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        
        # Add to current chat
        self.current_chat.append(f"{sender}: {message}")
    
    def start_new_chat(self):
        if self.current_chat:
            self.save_current_chat()
        self.current_chat = []
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.display_message("Bot: Started a new chat! Ask me anything from my knowledge base.", "bot")
    
    def save_current_chat(self):
        if self.current_chat:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            chat_data = {
                'title': self.current_chat[0][:50] + "..." if len(self.current_chat[0]) > 50 else self.current_chat[0],
                'timestamp': timestamp,
                'messages': self.current_chat
            }
            self.chat_history.append(chat_data)
            self.save_chat_history()
            self.update_history_list()
    
    def load_selected_chat(self, event):
        selection = self.history_listbox.curselection()
        if selection:
            index = selection[0]
            chat_data = self.chat_history[index]
            
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete(1.0, tk.END)
            
            for message in chat_data['messages']:
                if message.startswith('user:'):
                    self.chat_display.insert(tk.END, f"{message}\n\n", 'user')
                else:
                    self.chat_display.insert(tk.END, f"{message}\n\n", 'bot')
            
            self.chat_display.config(state=tk.DISABLED)
            self.chat_display.see(tk.END)
    
    def update_history_list(self):
        self.history_listbox.delete(0, tk.END)
        for chat in self.chat_history[-20:]:
            self.history_listbox.insert(tk.END, f"{chat['timestamp']} - {chat['title']}")
    
    def load_chat_history(self):
        try:
            if os.path.exists('chat_history.json'):
                with open('chat_history.json', 'r') as f:
                    self.chat_history = json.load(f)
                self.update_history_list()
        except:
            self.chat_history = []
    
    def save_chat_history(self):
        try:
            with open('chat_history.json', 'w') as f:
                json.dump(self.chat_history[-50:], f)
        except:
            pass
    
    def show_dataset_management(self):
        """Show dataset management window"""
        dataset_window = tk.Toplevel(self.root)
        dataset_window.title("Dataset Management")
        dataset_window.geometry("600x500")
        dataset_window.configure(bg='#2b2b2b')
        
        tk.Label(dataset_window, text="Dataset Management", font=('Arial', 16, 'bold'),
                bg='#2b2b2b', fg='white').pack(pady=20)
        
        # Dataset info
        info_text = f"Records: {len(self.dataset)}\nCategories: {len(self.dataset['category'].unique())}"
        info_label = tk.Label(dataset_window, text=info_text, font=('Arial', 12),
                             bg='#2b2b2b', fg='white')
        info_label.pack(pady=10)
        
        # Add new entry frame
        add_frame = tk.Frame(dataset_window, bg='#2b2b2b')
        add_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(add_frame, text="Add New Q&A:", bg='#2b2b2b', fg='white').pack(anchor=tk.W)
        
        # Question input
        tk.Label(add_frame, text="Question:", bg='#2b2b2b', fg='white').pack(anchor=tk.W)
        self.new_question = tk.Entry(add_frame, width=50, bg='#3c3c3c', fg='white')
        self.new_question.pack(fill=tk.X, pady=5)
        
        # Answer input
        tk.Label(add_frame, text="Answer:", bg='#2b2b2b', fg='white').pack(anchor=tk.W)
        self.new_answer = tk.Entry(add_frame, width=50, bg='#3c3c3c', fg='white')
        self.new_answer.pack(fill=tk.X, pady=5)
        
        # Category input
        tk.Label(add_frame, text="Category:", bg='#2b2b2b', fg='white').pack(anchor=tk.W)
        self.new_category = tk.Entry(add_frame, width=50, bg='#3c3c3c', fg='white')
        self.new_category.pack(fill=tk.X, pady=5)
        
        # Add button
        tk.Button(add_frame, text="Add to Dataset", command=self.add_to_dataset,
                 bg='#4CAF50', 
