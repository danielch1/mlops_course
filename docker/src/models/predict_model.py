import random

# Read the nice messages from the previous file
with open("data/interim/nice_messages.txt", "r") as file:
    nice_messages = file.readlines()

# Generate a random number
random_number = random.randint(1, 100)

# Select a random nice message from the list
selected_message = random.choice(nice_messages).strip()

# Create a response message that combines the nice message and random number
response_message = f"""Based on trained model, your prediction is: {selected_message}, {random_number}\n"""

# Specify the name of the output file
output_file_name = "data/interim/response_message.txt"

# Write the response message to the output file
with open(output_file_name, "w") as file:
    file.write(response_message)

print(f"Response message has been saved to {output_file_name}")
