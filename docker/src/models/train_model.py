nice_messages = [
    "You are amazing!",
    "Keep up the great work!",
    "You make the world a better place.",
    "Believe in yourself and your abilities.",
    "You deserve all the happiness in the world.",
]

# Specify the name of the text file
file_name = "data/interim/nice_messages.txt"

# Open the file in write mode and write the messages
with open(file_name, "w") as file:
    file.write("I've created a dummy, but nice model!" + "\n")
    for message in nice_messages:
        file.write(message + "\n")

print(f"Nice messages have been saved to {file_name}")