# Define a dictionary to store the rules and responses
rules = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi! What's on your mind?",
    "how are you": "I'm doing great, thanks! How about you?",
    "what is your name": "My name is Chatty, nice to meet you!",
    "goodbye": "Goodbye! It was nice chatting with you.",
    "default": "I didn't understand that. Can you please rephrase?"
}

def chatbot(user_input):
    # Convert the user input to lowercase for case-insensitive matching
    user_input = user_input.lower()

    # Check if the user input matches any of the predefined rules
    for rule in rules:
        if rule in user_input:
            return rules[rule]

    # If no rule matches, return the default response
    return rules["default"]

# Test the chatbot
while True:
    user_input = input("You: ")
    response = chatbot(user_input)
    print("Chatty:", response)
